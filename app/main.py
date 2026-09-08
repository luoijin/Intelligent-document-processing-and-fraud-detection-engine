from fastapi import FastAPI

app = FastAPI(
    title="IDPAFDE — Intelligent Document Processing & Fraud Detection Engine",
    version="0.1.0",
    description="""
    Portfolio-grade document processing pipeline that ingests receipt images,
    extracts structured data via CV/NLP, and flags anomalous/fraudulent records.
    """,
)


@app.get("/health", include_in_schema=False)
async def health():
    """Liveness/readiness check for deployment."""
    return {"status": "ok", "models_loaded": True}