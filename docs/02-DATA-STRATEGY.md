# Data Strategy

## 1. Free Dataset Sources (v1: receipts)

| Dataset | Content | License / Access | Notes |
|---|---|---|---|
| SROIE (ICDAR 2019) | ~1,000 scanned receipts with OCR + key-field annotations | Free, research use | Best starting point — already has ground-truth labels for merchant/date/total |
| Kaggle "Receipt / Invoice OCR" sets | Varied receipt photos | Free w/ Kaggle account | Use for augmenting image diversity (lighting, crumpling) |
| Self-collected | Photos of your own receipts | Free | Good for testing real-world noise; strip personal payment data before use |

No dataset in this plan requires payment or a commercial license for
non-commercial/portfolio use. Re-check each dataset's license before any
commercial use.

## 2. Target Schema (structured extraction output)

```json
{
  "document_id": "uuid",
  "merchant_name": "string",
  "date": "YYYY-MM-DD",
  "total_amount": "number",
  "currency": "string",
  "line_items": [
    {"description": "string", "quantity": "number", "unit_price": "number"}
  ],
  "ocr_confidence_avg": "float 0-1",
  "extraction_confidence": "float 0-1"
}
```

## 3. Labeling Plan

1. Start with SROIE's existing labels — no manual work needed for the
   first baseline.
2. For your own collected receipts, manually label 100–300 documents
   using a simple spreadsheet or a free tool (e.g., Label Studio,
   self-hosted, free).
3. Minimum fields to label per document: `merchant_name`, `date`,
   `total_amount`. Line items are a stretch label.
4. Track inter-field label confidence informally — if you're unsure of a
   value, mark it `uncertain` rather than guessing, so it can be excluded
   from training.

## 4. Synthetic Fraud Injection (for the anomaly model)

Since real fraud-labeled receipts are not available for free, generate
synthetic anomalies from your clean extracted dataset:

| Anomaly type | Injection method |
|---|---|
| Amount inflation | Multiply `total_amount` by 5–20x on a random sample |
| Duplicate submission | Duplicate a record with a different `document_id` |
| Date manipulation | Shift date outside plausible range (future-dated, decade-old) |
| Merchant mismatch | Swap `merchant_name` between unrelated records |
| Round-number stuffing | Force suspiciously round totals (e.g., exactly 1000.00) at higher frequency than natural distribution |

Label injected records as `is_synthetic_fraud = true` — this becomes the
target for supervised evaluation, while the Isolation Forest baseline
remains unsupervised.

## 5. Data Splits

- 70% train / 15% validation / 15% test, split **before** any synthetic
  fraud injection to avoid leakage.
- Keep a small "real-world holdout" (your own photographed receipts) that
  is never used in training — this is your true generalization check.

## 6. Data Storage

- Raw images: local filesystem (`/data/raw/`), never committed to git.
- Extracted structured data + labels: SQLite (`data.db`), version the
  schema, not the data, in git.
- Add a `.gitignore` entry for `/data/` to keep the repo free-tier-hostable
  and avoid committing any personal receipt images.
