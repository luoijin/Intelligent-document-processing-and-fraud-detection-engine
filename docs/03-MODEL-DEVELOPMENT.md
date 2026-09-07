# Model Development Plan

## 1. OCR Layer

| Option | Cost | When to use |
|---|---|---|
| Tesseract | Free, open-source, local | Fastest to integrate; use first |
| EasyOCR | Free, open-source, local, GPU-optional | Better accuracy on varied fonts/lighting than Tesseract |
| TrOCR (HuggingFace) | Free weights, needs more compute | v2 stretch, better on noisy/handwritten text |

**Decision for v1:** EasyOCR. No fine-tuning required — treat as a fixed
component and measure its raw output quality before investing further.

## 2. Entity Extraction Layer

### 2.1 Baseline (build first)
Rule-based / regex extraction over OCR tokens using positional heuristics:
- `total_amount`: largest currency-formatted number near the words
  "total"/"amount due".
- `date`: first token matching common date regex patterns.
- `merchant_name`: largest/topmost text block (often the header).

This baseline has no training cost and gives you a number to beat.

### 2.2 Trained Model
- **Model:** LayoutLMv3-base (HuggingFace, free/open weights).
- **Fine-tuning target:** token classification over OCR tokens with
  positional embeddings, predicting field labels
  (`O`, `B-MERCHANT`, `B-DATE`, `B-TOTAL`, etc.).
- **Compute:** fine-tune on Google Colab free tier (T4 GPU) — sufficient
  for a few hundred labeled receipts at small batch size.
- **Exit criteria:** field-level F1 on validation set exceeds the
  rule-based baseline by a documented margin.

## 3. Anomaly / Fraud Detection Layer

### 3.1 Feature Engineering
| Feature | Description |
|---|---|
| `amount_zscore` | Deviation of `total_amount` from that merchant's historical mean |
| `is_round_number` | Boolean, total ends in `.00` and > threshold |
| `days_since_last_from_merchant` | Recency pattern |
| `duplicate_hash_flag` | Fuzzy-matched near-duplicate of an existing record |
| `ocr_confidence_avg` | Low OCR confidence can correlate with tampering or poor scans |
| `date_plausibility` | Future-dated or implausibly old |

### 3.2 Models
| Model | Type | Cost | Role |
|---|---|---|---|
| Isolation Forest (scikit-learn) | Unsupervised | Free | Baseline anomaly score, no labels needed |
| XGBoost | Supervised | Free (open-source) | Trained on synthetic-fraud-labeled data once available |
| Autoencoder (stretch) | Unsupervised, deep | Free (PyTorch) | v2 comparison against Isolation Forest |

**Exit criteria:** on the synthetic-fraud validation set, the supervised
model's precision/recall is measured and compared against the Isolation
Forest baseline; document trade-offs (false positive rate matters more
than raw accuracy in a fraud-review context).

## 4. Model Versioning

- Track experiments with a free tool: MLflow (self-hosted, free) or even
  a simple `experiments.csv` log (model name, params, dataset version,
  metrics, date) if MLflow is overkill for solo use.
- Save model artifacts locally (`/models/{name}/{version}/`), never
  committed to git if large — use Git LFS (free tier) or exclude and
  document how to regenerate.

## 5. Compute Budget (all free-tier)

| Task | Free resource |
|---|---|
| LayoutLM fine-tuning | Google Colab free T4 GPU (session-limited) |
| Isolation Forest / XGBoost training | Local CPU — trivial cost |
| Experiment tracking | Self-hosted MLflow or local CSV |
| Batch inference testing | Local CPU/GPU or Kaggle free notebook GPU |
