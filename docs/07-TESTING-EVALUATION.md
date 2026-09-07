# Testing & Evaluation

## 1. Evaluation Metrics by Component

| Component | Metric | Target (v1) |
|---|---|---|
| OCR | Character Error Rate (CER) vs. SROIE ground truth text | Documented, no fixed target — establishes baseline |
| Baseline extractor | Field-level accuracy (exact match) for merchant/date/total | Documented, becomes benchmark |
| Trained extractor (LayoutLM) | Field-level F1 | Must exceed baseline accuracy |
| Anomaly model (Isolation Forest) | Precision/Recall on synthetic fraud set | Documented |
| Anomaly model (XGBoost) | Precision/Recall on synthetic fraud set | Must be compared against Isolation Forest |
| End-to-end pipeline | Latency (p95) | < 5s per document, local CPU |

## 2. Test Types

### 2.1 Unit Tests
- Preprocessing functions (deskew output shape/format).
- Feature builder (each engineered feature on known-input fixtures).
- API request/response schema validation.

### 2.2 Integration Tests
- Full pipeline on a fixed set of sample receipts, asserting the response
  shape and that no exceptions occur.

### 2.3 Model Evaluation (not classic unit tests, but tracked the same way)
- Run held-out test-set evaluation as part of CI whenever a model
  artifact changes, and record metrics to `experiments.csv` or MLflow.

### 2.4 Regression Guard
- Before replacing a model version, re-run it against the same held-out
  test set as the previous version and require the new metric to be
  greater-than-or-equal, or document why a regression is acceptable
  (e.g., large speed gain).

## 3. QA Checklist (run before calling a phase "done")

- [ ] Pipeline runs on a clean checkout (`docker compose up`) with no
      manual setup steps beyond documented `.env` configuration.
- [ ] All exit criteria in `06-ROADMAP-MILESTONES.md` for the current
      phase are met and the metric is written down somewhere durable
      (README, experiments log, or this doc).
- [ ] No paid API keys or paid services required anywhere in the run path.
- [ ] Error responses (malformed image, missing fields) return sane HTTP
      codes and messages, not stack traces.
- [ ] Personal data (your own receipts, if used) is excluded from
      anything committed to git.

## 4. Known Limitations to Document (don't hide these — name them)

- v1 supports receipts only; invoices/forms are out of scope.
- No authentication on the API — not production-secure as-is.
- Synthetic fraud is not a substitute for real fraud data; precision/
  recall numbers describe performance on synthetic patterns only.
- Free-tier hosting (if used) has cold-start latency and storage caps.
