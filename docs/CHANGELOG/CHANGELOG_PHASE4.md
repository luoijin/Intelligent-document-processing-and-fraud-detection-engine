# Changelog — Phase 4: Anomaly / Fraud Layer

**Roadmap reference:** `docs/06-ROADMAP-MILESTONES.md`, Phase 4
**Exit criteria (roadmap):** "Precision/recall on synthetic fraud
validation set documented for both models."
**Status:** ✅ Met — see [Validation](#validation) below.

---

## Summary

Implements the Feature Builder (`app/features.py`) and Anomaly/Fraud
Model (`app/anomaly.py`) components per `docs/01-ARCHITECTURE.md` §2.4-2.5,
plus the synthetic fraud dataset (`scripts/build_fraud_dataset.py`) and
training/evaluation scripts they need. Both Isolation Forest
(unsupervised baseline) and XGBoost (supervised) are trained and
compared, per `docs/03-MODEL-DEVELOPMENT.md` §3.2.

This phase does not depend on Phase 3 (trained extractor) — the
Feature Builder consumes structured records (`merchant_name`, `date`,
`total_amount`, `ocr_confidence_avg`), regardless of whether they came
from the Phase 2 rule-based extractor or a future Phase 3 model. It was
built and validated while Phase 3 remains scaffolded-but-not-executed
(see `CHANGELOG_PHASE3.md`) without violating phase-ordering — Phase 4
genuinely doesn't sit downstream of Phase 3 in the architecture
diagram's dependency sense, only in the roadmap's *listed* order.

## Data (leakage-safety notes)

Per `docs/02-DATA-STRATEGY.md` §4-5:

1. `scripts/build_fraud_dataset.py` generates 900 clean synthetic
   transactions across 10 merchants (this stands in for "records that
   came out of the extractor" — no images/OCR involved, consistent
   with Phase 4 being image-agnostic).
2. **Split 70/15/15 BEFORE any fraud injection** (train=630, val=135,
   test=135 before injection; see script for exact post-injection
   counts, which grow slightly due to the duplicate-submission method
   adding rows).
3. Fraud is injected **independently within each split** — a
   duplicate-submission or amount-inflation record generated from a
   train-split record can never land in val/test.
4. Merchant `mean`/`std` statistics (used for `amount_zscore`) are
   computed **once, from only the clean pre-injection train split**,
   then frozen and reused for val/test feature-building — this
   mirrors how a live system would use frozen historical stats on new
   incoming records, and prevents injected fraud amounts from
   corrupting what's considered "normal" for a merchant.

## Feature Builder (`app/features.py`)

Implements all 6 features from `docs/03-MODEL-DEVELOPMENT.md` §3.1:
`amount_zscore`, `is_round_number`, `days_since_last_from_merchant`,
`duplicate_hash_flag`, `ocr_confidence_avg`, `date_plausibility`.
`FEATURE_COLUMNS` is the stable, ordered interface between this module
and `app/anomaly.py` — both training (`scripts/build_fraud_dataset.py`)
and serving (Phase 5's `app/main.py`) go through
`build_feature_vector()` so column order can never drift between train
and inference.

Documented v1 defaults not specified in the plan (named here rather
than silently chosen):
- `date_plausibility`: linear ramp, fully implausible beyond 1 day
  future or 2 years past — no fixed value was given in the docs.
- Cold-start `amount_zscore` (merchant never seen before): `0.0`
  (treated as "not anomalous by amount," since there's no history to
  deviate from).

## Anomaly Model (`app/anomaly.py`)

- **Isolation Forest**: `n_estimators=200`, `contamination` set to the
  measured training-set fraud rate (0.094) — the one place the
  "unsupervised" model is told anything about the label distribution;
  it never sees `y_train` itself.
- **XGBoost**: `n_estimators=200`, `max_depth=4`,
  `scale_pos_weight` set from the train class imbalance.
- Both expose `score_*(model, X) -> np.ndarray` in `[0, 1]`
  (higher = more anomalous) so Phase 5's serving code doesn't need to
  know which model is loaded.
- `best_threshold()` sweeps thresholds for each model's best-F1
  operating point, for *documentation* purposes only — the threshold
  actually used in production (Phase 5) is chosen separately and more
  conservatively, since per §3.2 "false positive rate matters more
  than raw accuracy in a fraud-review context."

## Files Added

| File | Purpose |
|---|---|
| `app/features.py` | Feature Builder |
| `app/anomaly.py` | Isolation Forest + XGBoost training/scoring/eval |
| `scripts/build_fraud_dataset.py` | Synthetic clean transactions → split → fraud injection → feature CSVs |
| `scripts/train_anomaly_models.py` | Trains + saves both models from `data/fraud/train.csv` |
| `tests/test_phase4.py` | Exit-criteria test: precision/recall for both models on `data/fraud/val.csv`, recorded to `experiments.csv` |
| `docs/CHANGELOG/CHANGELOG_PHASE4.md` | This file |

Not committed (gitignored, regenerable): `data/fraud/*.csv`,
`models/anomaly/v1/*.joblib`.

## Known Gaps

- `duplicate_hash_flag` uses an exact-match key
  (`merchant|date|rounded_total`) as documented in
  `docs/02-DATA-STRATEGY.md` §4 ("Duplicate submission"), not true
  fuzzy matching — a near-duplicate with a 1-cent difference or a
  1-day date shift won't be flagged. Documented as a v1 narrowing, not
  a silent gap.
- The synthetic corpus's per-merchant amount distributions are Gaussian
  by construction — real receipt amounts are typically right-skewed.
  The z-score feature will behave somewhat differently on real data.
- 10 synthetic merchants only; no "brand-new merchant" cold-start case
  is represented in training (though the code path exists — see
  `COLD_START_ZSCORE` in `app/features.py`).

## Validation

```
$ python scripts/build_fraud_dataset.py
train: n=640, fraud=60 (9.4%)
val: n=137, fraud=12 (8.8%)
test: n=137, fraud=12 (8.8%)

$ python scripts/train_anomaly_models.py
Training on 640 records (60 fraud, contamination=0.094)
Saved models to models/anomaly/v1

$ python tests/test_phase4.py
Validation set: n=137, fraud=12 (8.8%)

Isolation Forest (unsupervised baseline):
  precision=1.000  recall=0.500  f1=0.667  (threshold=0.60)

XGBoost (supervised):
  precision=0.778  recall=0.583  f1=0.667  (threshold=0.25)

Comparison: XGBoost and Isolation Forest tie on F1.
🎉 Phase 4 exit criteria met.
```

**Result interpretation:** the two models tie on best-F1 but land at
different operating points — Isolation Forest is perfectly precise but
catches only half the injected fraud at its best threshold; XGBoost
catches more (58%) at the cost of some false positives. Per
`docs/03-MODEL-DEVELOPMENT.md` §3.2's own framing, this is exactly the
kind of trade-off write-up the exit criterion asks for, not a
"which one wins" verdict — XGBoost does **not** clearly beat the
baseline here, and that's recorded as-is in `experiments.csv` rather
than reframed as a win.

## Next Phase

Phase 5 wires this anomaly layer (plus Phase 1/2 OCR+extraction) behind
`POST /v1/documents` — see `CHANGELOG_PHASE5.md`.
