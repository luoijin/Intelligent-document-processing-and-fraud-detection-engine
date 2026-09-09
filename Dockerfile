FROM python:3.12-slim

WORKDIR /app

# Install runtime deps: curl (healthcheck), build-essential (native
# wheels), tesseract-ocr (Phase 1 OCR engine), libgl1 (OpenCV runtime
# dependency for headless image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY app/ ./app/

# Health check — validates the FastAPI /health endpoint
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]