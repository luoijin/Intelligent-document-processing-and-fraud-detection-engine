import datetime as dt
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from app import store
from app.anomaly import load_model, score_isolation_forest
from app.extraction import extract_fields
from app.extraction import to_dict as fields_to_dict
from app.features import FEATURE_COLUMNS, build_feature_vector, compute_merchant_stats
from app.ocr import run_ocr, to_dict

app = FastAPI(
    title="IDPAFDE — Intelligent Document Processing & Fraud Detection Engine",
    version="0.5.0",
    description="""
    Portfolio-grade document processing pipeline that ingests receipt images,
    extracts structured data via CV/NLP, and flags anomalous/fraudulent records.
    """,
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

# Phase 5 (docs/06-ROADMAP-MILESTONES.md): production anomaly model
# choice. Isolation Forest is used here rather than XGBoost — per
# CHANGELOG_PHASE4.md, the two tied on F1 (0.667) at their respective
# best thresholds, but Isolation Forest was perfectly precise
# (1.000 vs XGBoost's 0.778) at the cost of lower recall (0.500 vs
# 0.583). docs/03-MODEL-DEVELOPMENT.md §3.2: "false positive rate
# matters more than raw accuracy in a fraud-review context" — a
# missed fraud case still surfaces on manual audit later, but a
# wrongly-flagged legitimate receipt burns a reviewer's time on every
# single occurrence. This is a documented choice, not a silent
# default; revisit if XGBoost's precision improves with more/real
# training data.
ANOMALY_MODEL_PATH = Path(__file__).parent.parent / "models" / "anomaly" / "v1" / "isolation_forest.joblib"

# Review threshold: matches the best-F1 operating point Isolation
# Forest found on the Phase 4 synthetic validation set
# (see experiments.csv). A documented v1 default pending real-world
# tuning once real (non-synthetic) fraud outcomes are observed — see
# docs/07-TESTING-EVALUATION.md §4's "synthetic fraud is not a
# substitute for real fraud data" caveat.
REVIEW_THRESHOLD = 0.6

_anomaly_model = None


@app.on_event("startup")
async def on_startup():
    global _anomaly_model
    store.init_db()
    if ANOMALY_MODEL_PATH.exists():
        _anomaly_model = load_model(ANOMALY_MODEL_PATH)
    # If the model isn't there (fresh checkout, Phase 4 scripts not yet
    # run), _anomaly_model stays None and POST /v1/documents returns a
    # 503 explaining why, rather than crashing the whole app — the
    # OCR/extraction-only endpoints (Phase 1/2) still work without it.


@app.get("/health", include_in_schema=False)
async def health():
    """Liveness/readiness check for deployment."""
    return {"status": "ok", "models_loaded": _anomaly_model is not None}


@app.post("/v1/ocr/extract", tags=["Phase 1 — Vision Extraction"])
async def extract_raw_text(file: UploadFile = File(...)):
    """Phase 1 endpoint: preprocess + OCR a document image, returning raw
    text tokens with bounding boxes and confidence scores.

    This is intentionally NOT the full `/v1/documents` contract described
    in docs/04-API-SPEC.md — that endpoint (structured field extraction +
    anomaly scoring) lands in Phase 3/Phase 5 once the entity extractor
    and anomaly model exist. This endpoint exposes only what Phase 1
    implements: raw vision extraction.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported content type '{file.content_type}'. "
                   f"Expected one of {sorted(ALLOWED_CONTENT_TYPES)}.",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "").suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = run_ocr(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return to_dict(result)


@app.post("/v1/extraction/baseline", tags=["Phase 2 — Baseline Structured Extraction"])
async def extract_baseline_fields(file: UploadFile = File(...)):
    """Phase 2 endpoint: preprocess + OCR a document image, then run the
    rule-based baseline extractor over the OCR tokens to produce
    structured fields (`merchant_name`, `date`, `total_amount`).

    Like `/v1/ocr/extract`, this is intentionally NOT the full
    `POST /v1/documents` contract in `docs/04-API-SPEC.md` — no
    anomaly score, no persistence, no `document_id`. Those require the
    Phase 4 anomaly model and Phase 5 service wrapper. This endpoint
    exposes only what Phase 2 implements: the baseline structured
    extractor, layered on top of the Phase 1 OCR output.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported content type '{file.content_type}'. "
                   f"Expected one of {sorted(ALLOWED_CONTENT_TYPES)}.",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "").suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        ocr_result = run_ocr(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    fields = extract_fields(ocr_result)
    response = fields_to_dict(fields)
    response["ocr_confidence_avg"] = round(ocr_result.avg_confidence, 2)
    return response


def _save_upload_to_tmp(file: UploadFile) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "").suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return Path(tmp.name)


@app.post("/v1/documents", tags=["Phase 5 — Full Service"])
async def create_document(file: UploadFile = File(...)):
    """Phase 5 endpoint (docs/04-API-SPEC.md `POST /documents`): the
    full pipeline end-to-end — OCR (Phase 1) -> structured extraction
    (Phase 2 baseline) -> feature building + anomaly scoring (Phase 4)
    -> persistence (Result Store) -> response.

    DOCUMENTED DEVIATION from docs/06-ROADMAP-MILESTONES.md's Phase 5
    description ("Wire Phases 1-4"): this wires Phase 2's rule-based
    extractor, not a Phase 3 trained model — Phase 3 remains
    scaffolded-but-not-executed (see CHANGELOG_PHASE3.md; blocked on
    GPU/network access this environment doesn't have, not skipped by
    choice). Swapping in a Phase 3 model later requires no change here
    beyond the `extract_fields` import, since both return the same
    `ExtractedFields` shape.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported content type '{file.content_type}'. "
                   f"Expected one of {sorted(ALLOWED_CONTENT_TYPES)}.",
        )
    if _anomaly_model is None:
        raise HTTPException(
            status_code=503,
            detail="Anomaly model not loaded — run "
                   "'python scripts/build_fraud_dataset.py && "
                   "python scripts/train_anomaly_models.py' first "
                   "(see docs/CHANGELOG/CHANGELOG_PHASE4.md).",
        )

    tmp_path = _save_upload_to_tmp(file)
    try:
        ocr_result = run_ocr(str(tmp_path))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

    fields = extract_fields(ocr_result)
    ocr_confidence_avg = round(ocr_result.avg_confidence, 2)

    # --- Feature building (Phase 4), using LIVE history from the
    # Result Store rather than the frozen Phase 4 training stats — a
    # fresh database starts with no merchant history at all, which
    # exercises the cold-start path in app/features.py by design
    # (this is what a real new deployment looks like on day one).
    record = {
        "merchant_name": fields.merchant_name or "UNKNOWN",
        "date": fields.date or dt.date.today().isoformat(),
        "total_amount": fields.total_amount if fields.total_amount is not None else 0.0,
        "ocr_confidence_avg": ocr_confidence_avg,
    }

    prior_records = store.get_prior_records_for_merchant(record["merchant_name"])
    merchant_stats = compute_merchant_stats(prior_records) if prior_records else {}
    if prior_records:
        last_date = dt.date.fromisoformat(prior_records[-1]["date"])
        record["days_since_last_from_merchant"] = (
            dt.date.fromisoformat(record["date"]) - last_date
        ).days
    else:
        record["days_since_last_from_merchant"] = None

    prior_hashes = store.get_all_record_hashes()
    features = build_feature_vector(record, merchant_stats, prior_hashes)
    X = np.array([[features[c] for c in FEATURE_COLUMNS]])

    anomaly_score = float(score_isolation_forest(_anomaly_model, X)[0])
    review_required = anomaly_score >= REVIEW_THRESHOLD

    document_id = str(uuid.uuid4())
    from app.features import _record_hash  # local import: internal helper, not part of the public feature-builder API
    persisted = {
        "document_id": document_id,
        "merchant_name": record["merchant_name"],
        "date": record["date"],
        "total_amount": record["total_amount"],
        "currency": fields.currency,
        "extraction_confidence": fields.extraction_confidence,
        "ocr_confidence_avg": ocr_confidence_avg,
        "anomaly_score": round(anomaly_score, 4),
        "review_required": int(review_required),
        "record_hash": _record_hash(record),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    store.insert_document(persisted)

    return {
        "document_id": document_id,
        "status": "processed",
        "fields": {
            "merchant_name": fields.merchant_name,
            "date": fields.date,
            "total_amount": fields.total_amount,
            "currency": fields.currency,
        },
        "extraction_confidence": fields.extraction_confidence,
        "anomaly_score": round(anomaly_score, 4),
        "review_required": review_required,
    }


@app.get("/v1/documents/{document_id}", tags=["Phase 5 — Full Service"])
async def get_document(document_id: str):
    """docs/04-API-SPEC.md `GET /documents/{document_id}`. Returns the
    persisted record for a previously processed document."""
    row = store.get_document(document_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No document with id '{document_id}'.")

    return {
        "document_id": row["document_id"],
        "status": "processed",
        "fields": {
            "merchant_name": row["merchant_name"],
            "date": row["date"],
            "total_amount": row["total_amount"],
            "currency": row["currency"],
        },
        "extraction_confidence": row["extraction_confidence"],
        "anomaly_score": row["anomaly_score"],
        "review_required": bool(row["review_required"]),
    }