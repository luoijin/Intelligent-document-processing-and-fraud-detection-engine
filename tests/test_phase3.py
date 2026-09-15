#!/usr/bin/env python3
"""
Phase 3 exit-criteria validation, per docs/06-ROADMAP-MILESTONES.md:

    "Exit criteria: fine-tuned model's F1 documented and compared
    against the Phase 2 baseline, with the comparison written down
    (even if the trained model doesn't win — that's a valid,
    documentable result)."

This test does NOT train anything (see scripts/train_layoutlmv3.py,
which must be run first — on Colab per docs/03-MODEL-DEVELOPMENT.md
§5, not locally). It only evaluates an already-fine-tuned checkpoint
at models/layoutlmv3/v1/, computes field-level F1 against
data/layoutlm/train.jsonl's labels, and appends the comparison to
experiments.csv next to the Phase 2 row.

If no checkpoint exists, this test reports that plainly and exits
non-zero — the Phase 3 exit criterion is NOT met yet, and this script
will not invent a number to make it look otherwise (see project
instructions: "Do not invent metrics, dataset sizes, or results").

Run: python tests/test_phase3.py
"""
import csv
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

MODEL_DIR = REPO / "models" / "layoutlmv3" / "v1"
DATA_PATH = REPO / "data" / "layoutlm" / "val.jsonl"
IMAGE_DIR = REPO / "data" / "sample_receipts"
EXPERIMENTS_LOG = REPO / "experiments.csv"

LABELS = [
    "O",
    "B-MERCHANT", "I-MERCHANT",
    "B-DATE", "I-DATE",
    "B-TOTAL", "I-TOTAL",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}


def check_prerequisites() -> bool:
    ok = True
    if not DATA_PATH.exists():
        print(f"❌ No labeled data at {DATA_PATH}. "
              f"Run: python scripts/prepare_layoutlm_labels.py")
        ok = False
    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        print(f"❌ No fine-tuned model at {MODEL_DIR}. "
              f"Run: python scripts/train_layoutlmv3.py "
              f"(on Colab free-tier GPU per docs/03-MODEL-DEVELOPMENT.md §5 "
              f"— this has not been executed yet; see "
              f"docs/CHANGELOG/CHANGELOG_PHASE3.md).")
        ok = False
    return ok


def get_phase2_baseline() -> float | None:
    if not EXPERIMENTS_LOG.exists():
        return None
    with open(EXPERIMENTS_LOG) as f:
        rows = list(csv.DictReader(f))
    phase2_rows = [r for r in rows if "Phase 2" in r.get("model", "")]
    if not phase2_rows:
        return None
    return float(phase2_rows[-1]["overall_field_accuracy"])


def run_evaluation():
    """Load the fine-tuned checkpoint, run it over data/layoutlm/train.jsonl,
    compute field-level (entity-level) F1 with seqeval. Returns None if
    dependencies are missing."""
    try:
        import torch
        from PIL import Image
        from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
        from seqeval.metrics import f1_score, classification_report
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install -r requirements-phase3.txt")
        return None

    processor = LayoutLMv3Processor.from_pretrained(str(MODEL_DIR), apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    records = [json.loads(l) for l in open(DATA_PATH) if l.strip()]

    all_true, all_pred = [], []
    for rec in records:
        image = Image.open(IMAGE_DIR / rec["image"]).convert("RGB")
        width, height = image.size
        boxes = [
            [int(1000 * b[0] / width), int(1000 * b[1] / height),
             int(1000 * b[2] / width), int(1000 * b[3] / height)]
            for b in rec["bboxes"]
        ]
        encoding = processor(image, rec["tokens"], boxes=boxes,
                              truncation=True, padding="max_length",
                              return_tensors="pt")
        with torch.no_grad():
            logits = model(**encoding).logits
        predictions = logits.argmax(-1).squeeze().tolist()

        # LayoutLMv3's processor subdivides words into subword tokens and
        # pads; only the first subword of each original word carries a
        # meaningful label-aligned prediction. word_ids() maps encoded
        # positions back to original token indices.
        word_ids = encoding.word_ids(batch_index=0)
        seen = set()
        pred_labels, true_labels = [], []
        for idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen:
                continue
            seen.add(word_id)
            pred_labels.append(ID2LABEL[predictions[idx]])
            true_labels.append(rec["labels"][word_id])

        all_true.append(true_labels)
        all_pred.append(pred_labels)

    overall_f1 = f1_score(all_true, all_pred)
    report = classification_report(all_true, all_pred, zero_division=0)
    return overall_f1, report, len(records)


def record_experiment(f1: float, n_samples: int, baseline: float | None) -> None:
    is_new = not EXPERIMENTS_LOG.exists()
    with open(EXPERIMENTS_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow([
                "date", "model", "dataset", "n_samples",
                "merchant_name_acc", "date_acc", "total_amount_acc",
                "overall_field_accuracy", "notes",
            ])
        beat_baseline = (
            "beats Phase 2 baseline" if baseline is not None and f1 > baseline
            else "does NOT beat Phase 2 baseline" if baseline is not None
            else "no Phase 2 baseline found to compare against"
        )
        writer.writerow([
            dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "layoutlmv3_finetuned (Phase 3)",
            "silver_labeled_synthetic_sample_receipts (n={})".format(n_samples),
            n_samples,
            "n/a", "n/a", "n/a",
            f"{f1:.3f}",
            f"Entity-level F1 (seqeval). {beat_baseline}. "
            f"n={n_samples} is far below the 100-300 documents "
            f"docs/02-DATA-STRATEGY.md calls a usable minimum — "
            f"not a meaningful generalization estimate.",
        ])
    print(f"✅ Recorded metrics to {EXPERIMENTS_LOG}")


def main() -> None:
    print("🔎 Validating Phase 3 trained extraction model...\n")
    if not check_prerequisites():
        print("\n❌ Phase 3 exit criteria NOT met: no fine-tuned model to "
              "evaluate. See docs/CHANGELOG/CHANGELOG_PHASE3.md for status "
              "and next steps.")
        sys.exit(1)

    result = run_evaluation()
    if result is None:
        sys.exit(1)

    f1, report, n_samples = result
    baseline = get_phase2_baseline()

    print(f"Entity-level F1 (n={n_samples} samples): {f1:.3f}\n")
    print(report)
    if baseline is not None:
        verdict = "beats" if f1 > baseline else "does NOT beat"
        print(f"\nPhase 2 rule-based baseline: {baseline:.3f}")
        print(f"Phase 3 LayoutLMv3 F1:       {f1:.3f}  ({verdict} baseline)")
    else:
        print("\n⚠️  No Phase 2 baseline found in experiments.csv to compare against.")

    record_experiment(f1, n_samples, baseline)

    print(
        f"\n⚠️  n={n_samples} — this comparison is a pipeline validation, "
        f"not the real Phase 3 result. The exit criterion requires "
        f"comparing against a meaningful validation set (real SROIE or "
        f"100-300+ manually labeled documents per "
        f"docs/02-DATA-STRATEGY.md §3). Treat this run as proof the "
        f"train -> evaluate -> compare -> log pipeline works end to end, "
        f"then re-run against real data before calling Phase 3 done."
    )


if __name__ == "__main__":
    main()
