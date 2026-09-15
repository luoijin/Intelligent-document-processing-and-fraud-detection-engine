# Changelog — Phase 3: Trained Extraction Model

**Roadmap reference:** `docs/06-ROADMAP-MILESTONES.md`, Phase 3
**Exit criteria (roadmap):** "Fine-tuned model's F1 documented and
compared against the Phase 2 baseline, with the comparison written
down (even if the trained model doesn't win — that's a valid,
documentable result)."
**Status:** 🔄 Scaffolded, exit criteria **NOT met**. Do not treat
Phase 3 as complete — no model has actually been fine-tuned. See
"Blockers" below before "Next Steps."

---

## Summary

Phase 3 calls for fine-tuning LayoutLMv3-base (`docs/03-MODEL-DEVELOPMENT.md`
§2.2) and documenting its field-level F1 against the Phase 2 rule-based
baseline (`overall_field_accuracy=1.000` on synthetic data, recorded in
`experiments.csv`). This changelog entry covers the pipeline built to
do that — data labeling, training, evaluation — **not** an actual
completed training run. That distinction matters: this project's own
rules (see project instructions §7-8) prohibit invented metrics and
phase-skipping, and claiming Phase 3 "done" without a real fine-tuned
model would violate both.

## Blockers (why this wasn't executed)

The environment this scaffold was built in cannot run the actual
fine-tuning step:

1. **No GPU.** `nvidia-smi` is unavailable. CPU fine-tuning of a
   transformer isn't realistic and isn't what the plan prescribes —
   `docs/03-MODEL-DEVELOPMENT.md` §5 explicitly calls for Colab's free
   T4 GPU.
2. **No network access to `huggingface.co`.** The pretrained
   `microsoft/layoutlmv3-base` weights can't be downloaded from this
   sandbox (network egress is restricted to package registries —
   pypi/npm/crates/github — not model hubs).
3. **No real labeled training data.** Only the 3 synthetic samples
   from Phase 1/2 exist. `docs/02-DATA-STRATEGY.md` §3 calls 100-300
   manually labeled documents a usable minimum; 3 silver-labeled
   synthetic samples is a pipeline smoke test at best, not training
   data.

None of these are things to route around with a substitute (e.g., a
smaller open model that could plausibly be pulled from an allowed
domain, or fabricating an F1 number) — that would be a silent,
undocumented deviation from the plan and a fabricated result,
respectively, both of which the project instructions prohibit
outright. The correct move is what's below: build the pipeline, name
the blocker, hand it back for execution on Colab.

## What Was Built

| File | Purpose |
|---|---|
| `scripts/prepare_layoutlm_labels.py` | Converts Phase 1 OCR tokens + Phase 2 `ground_truth.json` into BIO token-classification labels (`O`, `B-/I-MERCHANT`, `B-/I-DATE`, `B-/I-TOTAL`) via exact-match silver labeling. Writes `data/layoutlm/train.jsonl`. |
| `scripts/train_layoutlmv3.py` | Fine-tunes `microsoft/layoutlmv3-base` as a token classifier on `data/layoutlm/train.jsonl`. Designed to run on Colab (T4), not locally — see in-file docstring. Saves to `models/layoutlmv3/v1/` (gitignored, per `docs/03-MODEL-DEVELOPMENT.md` §4). |
| `tests/test_phase3.py` | Exit-criteria test: loads a fine-tuned checkpoint from `models/layoutlmv3/v1/`, computes entity-level F1 (seqeval) on `data/layoutlm/train.jsonl`, compares against the Phase 2 baseline pulled from `experiments.csv`, and appends the comparison. **If no checkpoint exists, it fails loudly and exits non-zero rather than reporting a number.** |
| `requirements-phase3.txt` | Train-time-only deps (`torch`, `transformers`, `datasets`, `accelerate`, `seqeval`), deliberately kept out of `requirements.txt` — the served API stays on the light CPU-only OCR/extraction stack; these are only needed to run the two scripts above. |
| `docs/CHANGELOG/CHANGELOG_PHASE3.md` | This file. |

### Validation performed in this environment

```
$ python scripts/prepare_layoutlm_labels.py
  ✅ sample_receipt_01.png: 21 tokens, labels=['B-MERCHANT', 'I-MERCHANT', 'B-DATE', 'B-TOTAL']
  ✅ sample_receipt_02.png: 17 tokens, labels=['B-MERCHANT', 'I-MERCHANT', 'I-MERCHANT', 'B-DATE', 'B-TOTAL']
  ✅ sample_receipt_03.png: 17 tokens, labels=['B-MERCHANT', 'I-MERCHANT', 'B-DATE', 'B-TOTAL']
Wrote 3 labeled records to data/layoutlm/train.jsonl

$ python tests/test_phase3.py
🔎 Validating Phase 3 trained extraction model...
❌ No fine-tuned model at models/layoutlmv3/v1. Run: python scripts/train_layoutlmv3.py ...
❌ Phase 3 exit criteria NOT met: no fine-tuned model to evaluate.
(exit code 1)
```

The labeling step runs and produces correct BIO labels on all 3
synthetic samples (confirmed by inspection). `train_layoutlmv3.py` was
**not** run — it needs the GPU/network access this environment
doesn't have. `test_phase3.py` was run and confirmed to fail safely
(no model → no fabricated F1 → non-zero exit), which is itself the
behavior being validated here (see "Blockers").

## Known Gaps / Deliberate Non-Goals of This Scaffold

- The silver-labeler in `prepare_layoutlm_labels.py` does exact
  string-match alignment against clean synthetic OCR output. It will
  under-label noisy real receipts (OCR misreads mean the ground-truth
  string often won't appear verbatim) — it is not a substitute for the
  manual labeling `docs/02-DATA-STRATEGY.md` §3 calls for on real data.
- `train_layoutlmv3.py` has fixed hyperparameters (20 epochs, batch
  size 2, lr 5e-5) tuned for "does this run at all on 3 samples,"
  not for a real training set — expect to retune once real data
  exists.
- No train/val/test split exists yet for this data (n=3 is too small
  to split meaningfully). Per `docs/02-DATA-STRATEGY.md` §5, the real
  run needs the 70/15/15 split *before* evaluation, and `test_phase3.py`
  currently evaluates on the same data it would be trained on — that's
  acceptable only for the pipeline smoke test this is, and must not be
  read as a real F1.
- `currency` and `line_items` are out of scope for this label set,
  consistent with the Phase 2 baseline's scope.

## Next Steps (to actually close Phase 3)

1. Get real labeled data: either the real SROIE dataset (manual
   authenticated download, per the deviation already logged in
   `CHANGELOG_PHASE1.md`/`CHANGELOG_PHASE2.md`) or 100-300 manually
   labeled receipts per `docs/02-DATA-STRATEGY.md` §3.
2. Run `scripts/prepare_layoutlm_labels.py` against that data (it'll
   need adapting from exact-match silver-labeling to consuming real
   manual labels — the JSONL output shape can stay the same).
3. Split 70/15/15 before anything else touches the data.
4. Run `scripts/train_layoutlmv3.py` on Colab free-tier T4.
5. Run `tests/test_phase3.py` against the *validation* split (not the
   training data) to get a real, comparable F1.
6. Record whichever way the comparison goes — the roadmap explicitly
   treats "the trained model doesn't win" as a valid, documentable
   Phase 3 outcome. Only then update this changelog's Status line and
   the roadmap table in `README.md` to ✅.

## Files Not Modified

`app/*`, `main.py`, `docker-compose.yml`, `Dockerfile`,
`requirements.txt` — Phase 3's training pipeline is entirely offline
tooling (`scripts/`, `tests/test_phase3.py`) and does not touch the
served API surface. Wiring a trained model into `/v1/extraction/*`
happens in Phase 5's service wrapper per `docs/06-ROADMAP-MILESTONES.md`,
once there's an actual model artifact worth serving.
