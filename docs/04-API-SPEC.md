# API Specification

Framework: **FastAPI** (free, open-source, auto-generates OpenAPI docs at
`/docs`).

## 1. Endpoints

### `POST /documents`
Ingest a document image for processing.

**Request:** `multipart/form-data`, field `file` (image/jpeg, image/png)

**Response 200:**
```json
{
  "document_id": "uuid",
  "status": "processed",
  "fields": {
    "merchant_name": "string",
    "date": "YYYY-MM-DD",
    "total_amount": 42.50,
    "currency": "PHP"
  },
  "extraction_confidence": 0.87,
  "anomaly_score": 0.12,
  "review_required": false
}
```

**Response 422:** malformed/unreadable image.

---

### `GET /documents/{document_id}`
Retrieve a previously processed document's record.

**Response 200:** same shape as `POST /documents` response.

**Response 404:** unknown `document_id`.

---

### `GET /documents`
List processed documents with optional filters.

**Query params:** `review_required` (bool), `merchant_name` (string),
`date_from`, `date_to`, `limit`, `offset`.

**Response 200:** paginated array of document summaries.

---

### `POST /documents/{document_id}/review`
Submit a human correction for a flagged/low-confidence document.
Feeds the active-learning retraining dataset.

**Request:**
```json
{
  "corrected_fields": {"total_amount": 45.00},
  "reviewer_note": "OCR misread 4 as 4.5"
}
```

**Response 200:** confirmation + updated record.

---

### `GET /health`
Liveness/readiness check for deployment.

**Response 200:** `{"status": "ok", "models_loaded": true}`

## 2. Error Format (all endpoints)

```json
{
  "error": "string code",
  "message": "human-readable description"
}
```

## 3. Non-Functional Requirements

| Requirement | v1 Target |
|---|---|
| Latency (p95, single document) | < 5s on local CPU |
| Availability | Best-effort (single instance, free-tier host) |
| Auth | None required for v1 (local/demo use); note as a known gap |
| Rate limiting | Not required for v1; document as a v2 gap if publicly hosted |

## 4. Versioning

Prefix routes with `/v1/` from the start (`/v1/documents`, etc.) so a
breaking `/v2/` extraction schema can be introduced later without
breaking existing clients.
