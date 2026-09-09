import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.ocr import run_ocr, to_dict

app = FastAPI(
    title="IDPAFDE — Intelligent Document Processing & Fraud Detection Engine",
    version="0.2.0",
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