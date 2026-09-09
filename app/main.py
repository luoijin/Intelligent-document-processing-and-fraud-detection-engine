import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.extraction import extract_fields
from app.extraction import to_dict as fields_to_dict
from app.ocr import run_ocr, to_dict

app = FastAPI(
    title="IDPAFDE — Intelligent Document Processing & Fraud Detection Engine",
    version="0.3.0",
    description="""
    Portfolio-grade document processing pipeline that ingests receipt images,
    extracts structured data via CV/NLP, and flags anomalous/fraudulent records.
    """,
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


@app.get("/health", include_in_schema=False)
async def health():
    """Liveness/readiness check for deployment."""
    return {"status": "ok", "models_loaded": True}


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