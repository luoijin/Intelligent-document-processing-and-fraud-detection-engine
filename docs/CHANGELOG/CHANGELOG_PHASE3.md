# Changelog — Phase 3: Trained Extraction Model

**Roadmap reference:** `docs/06-ROADMAP-MILESTONES.md`, Phase 3
**Exit criteria (roadmap):** "Fine-tuned model's F1 documented and
compared against the Phase 2 baseline, with the comparison written
down (even if the trained model doesn't win — that's a valid,
documentable result)."
**Status:** ✅ Met — see [Result](#result) below. The fine-tuned model
does **not** beat the Phase 2 baseline; per the exit criteria above,
that is itself a valid, documented outcome, not a failure to close
the phase.

---

## Summary

LayoutLMv3-base was fine-tuned on real SROIE receipt data and
evaluated against a held-out validation split, with the result
compared against the Phase 2 rule-based baseline and logged to
`experiments.csv`. The original in-sandbox scaffold (see "Original
Scaffold" below) could not execute the fine-tuning step itself —
execution happened afterward, on Google Colab, as the scaffold
anticipated.

## Result

- **Data:** 626 real, labeled receipts pulled from the SROIE dataset
  via the Hugging Face `jsdnrs/ICDAR2019-SROIE` mirror (an
  authenticated manual SROIE download was not required this way).
  Split 70/15/15 → 438 train / 93 validation / 95 test (seeded shuffle,
  `random.seed(42)`). The test split has not been touched and is
  reserved for a final, only-look-once check.
- **Training:** `scripts/train_layoutlmv3.py` run on Google Colab
  (free-tier T4 GPU), fine-tuning `microsoft/layoutlmv3-base` for
  ~2.3 epochs (500 steps) on the 438-sample train split. Training
  loss fell from ~1.0 to ~0.06-0.09. Checkpoint saved to
  `models/layoutlmv3/v1/`.
- **Evaluation:** `tests/test_phase3.py`, run locally against the
  93-sample **validation** split (`data/layoutlm/val.jsonl`) — not
  the training data, so this is a genuine generalization estimate.

  ```
  Entity-level F1 (n=93 samples): 0.832

                precision    recall  f1-score   support

          DATE       0.92      0.97      0.95       109
      MERCHANT       0.86      0.80      0.83       140
         TOTAL       0.67      0.76      0.71        95

     micro avg       0.82      0.84      0.83       344
     macro avg       0.82      0.84      0.83       344
  weighted avg       0.83      0.84      0.83       344

  Phase 2 rule-based baseline: 0.889
  Phase 3 LayoutLMv3 F1:       0.832  (does NOT beat baseline)
  ```

- **Verdict:** the fine-tuned model does not beat the Phase 2
  rule-based baseline overall (0.832 vs. 0.889), but the per-field
  breakdown is the more useful finding: LayoutLMv3 is strong on DATE
  (0.95) and solid on MERCHANT (0.83), while TOTAL is the weak field
  (0.71) — plausibly because receipts often contain several numbers
  that look like a total (subtotal, tax, tip), which is harder for a
  token classifier to disambiguate than for a rule that looks for a
  line literally labeled "TOTAL." This is recorded in
  `experiments.csv` next to the Phase 2 row.
- **n=93 vs. the 100-300 minimum:** `docs/02-DATA-STRATEGY.md` §3
  calls 100-300 documents a usable minimum; 93 is close but slightly
  under it. `val.jsonl` + `test.jsonl` together (188 samples) would
  clear that bar with room, at the cost of spending the held-back
  test set now rather than saving it for a final check — judged not
  worth doing for this project, so the reported number stays the
  93-sample validation F1 above.

## Deviations from the Original Scaffold's Plan

- **Labeling path changed.** The original scaffold's
  `scripts/prepare_layoutlm_labels.py` (exact-match silver-labeling
  against clean synthetic OCR text) was not adapted for real data as
  originally planned. Instead, SROIE's own ground-truth fields
  (`company`/`date`/`total`) were BIO-tagged directly against SROIE's
  own OCR word/box arrays in a Colab notebook cell, using exact
  multi-token match with a token-overlap fallback for line-break
  mismatches. `prepare_layoutlm_labels.py` itself is unchanged and
  still only used for the original 3 synthetic samples.
- **`tests/test_phase3.py` was edited**, changing `DATA_PATH` from
  `data/layoutlm/train.jsonl` to `data/layoutlm/val.jsonl` — necessary
  so evaluation runs against held-out data rather than the same data
  the model trained on (train/test leakage). The docstring, an
  in-code comment, and the dataset label written to `experiments.csv`
  still say `train.jsonl`/`silver_labeled_synthetic_sample_receipts`
  in a couple of places and should be treated as stale text, not
  accurate of what actually ran.
- **Schema mismatch, resolved manually.** The Colab-side SROIE
  labeling cell wrote `image_file`/`ner_tags` (numeric label ids) per
  record; `tests/test_phase3.py` expects `image`/`labels` (string
  tags). Records were remapped to the expected field names/values
  before evaluation.

## Original Scaffold (historical — why Phase 3 wasn't executable in-sandbox)

The sandbox this project was originally scaffolded in could not run
the actual fine-tuning step: no GPU (`nvidia-smi` unavailable, and CPU
fine-tuning of a transformer isn't realistic or what the plan
prescribes — `docs/03-MODEL-DEVELOPMENT.md` §5 calls for Colab's free
T4 GPU specifically); no network access to `huggingface.co` to pull
`microsoft/layoutlmv3-base`; and no real labeled data beyond the 3
synthetic Phase 1/2 samples. Rather than route around that with a
substitute model or an invented F1, the scaffold below was built and
handed off for execution elsewhere — which is what subsequently
happened, on Colab, with real SROIE data, per "Result" above.

### What Was Built

| File | Purpose |
|---|---|
| `scripts/prepare_layoutlm_labels.py` | Converts Phase 1 OCR tokens + Phase 2 `ground_truth.json` into BIO labels via exact-match silver labeling, for the 3 synthetic samples only (see "Deviations" above — not what ended up producing the real training data). |
| `scripts/train_layoutlmv3.py` | Fine-tunes `microsoft/layoutlmv3-base` as a token classifier. Designed to run on Colab (T4), not locally. Saves to `models/layoutlmv3/v1/` (gitignored, per `docs/03-MODEL-DEVELOPMENT.md` §4). |
| `tests/test_phase3.py` | Exit-criteria test: loads the fine-tuned checkpoint, computes entity-level F1 (seqeval), compares against the Phase 2 baseline from `experiments.csv`, appends the comparison. Fails loudly with a non-zero exit if no checkpoint exists, rather than reporting a number. |
| `requirements-phase3.txt` | Train-time-only deps (`torch`, `transformers`, `datasets`, `accelerate`, `seqeval`), kept out of `requirements.txt` since the served API stays on the light CPU-only stack. |

### Smoke test performed in the original sandbox (n=3, superseded)

```
$ python tests/test_phase3.py   # against the 3-sample synthetic set, pre-SROIE
Entity-level F1 (n=3 samples): 0.300
        DATE      f1=0.86
    MERCHANT      f1=0.00
       TOTAL      f1=0.00
Phase 2 rule-based baseline: 0.889
Phase 3 LayoutLMv3 F1:       0.300  (does NOT beat baseline)
```

This confirmed the train → evaluate → compare → log pipeline worked
end to end, but was explicitly not a meaningful result (3 samples,
2 of 3 fields at 0.00 purely from too little data to learn from) —
superseded by the 93-sample real-data result above.

## Known Gaps

- `currency` and `line_items` remain out of scope for this label set,
  consistent with the Phase 2 baseline's scope.
- The trained model is not yet wired into the served API
  (`/v1/extraction/*` still runs the Phase 2 rule-based extractor).
  That integration is Phase 5's concern, not Phase 3's — see
  `CHANGELOG_PHASE5.md`.
- `tests/test_phase3.py`'s docstring, an in-code comment, and the
  dataset label it writes to `experiments.csv` still reference
  `train.jsonl` / "synthetic" in a few places despite now running
  against real SROIE `val.jsonl` — cosmetic staleness, not a
  functional bug, but worth cleaning up if this test is touched again.

## Files Not Modified

`app/*`, `main.py`, `docker-compose.yml`, `Dockerfile`,
`requirements.txt` — Phase 3 is entirely offline tooling (`scripts/`,
`tests/test_phase3.py`) and does not touch the served API surface.
Wiring a trained model into `/v1/extraction/*` happens in Phase 5's
service wrapper, per `docs/06-ROADMAP-MILESTONES.md`.
