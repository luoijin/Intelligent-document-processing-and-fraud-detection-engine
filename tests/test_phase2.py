#!/usr/bin/env python3
"""
Phase 2 exit-criteria validation, per docs/06-ROADMAP-MILESTONES.md:

    "Exit criteria: baseline field-level accuracy measured and recorded
    against SROIE ground truth (this number becomes your benchmark)."

Since the real SROIE dataset must be downloaded by the user (same
constraint documented for Phase 1 — see docs/CHANGELOG/CHANGELOG_PHASE1.md and CHANGELOG_PHASE2.md), this measures the rule-based extractor (app.extraction.extract_fields) against the synthetic ground_truth.json written by
scripts/generate_sample_receipts.py, as a stand-in. Swapping in real
SROIE images + ground truth requires no code change — only pointing
SAMPLE_DIR/ground truth at the new data.

Field-level accuracy (exact match per field) is computed for
merchant_name, date, and total_amount, then appended to
`experiments.csv` at the repo root so the metric is recorded somewhere
durable, per docs/07-TESTING-EVALUATION.md §2.3.

Run: python tests/test_phase2.py
"""
import csv
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

SAMPLE_DIR = REPO / "data" / "sample_receipts"
GROUND_TRUTH_PATH = SAMPLE_DIR / "ground_truth.json"
EXPERIMENTS_LOG = REPO / "experiments.csv"

FIELDS = ("merchant_name", "date", "total_amount")


def check_ground_truth_exists() -> bool:
    if not GROUND_TRUTH_PATH.exists():
        print(f"❌ No ground truth found at {GROUND_TRUTH_PATH}. "
              f"Run: python scripts/generate_sample_receipts.py")
        return False
    print(f"✅ Ground truth found at {GROUND_TRUTH_PATH}")
    return True


def _fields_match(predicted, expected, field: str) -> bool:
    if field == "total_amount":
        if predicted is None or expected is None:
            return predicted == expected
        return abs(float(predicted) - float(expected)) < 0.01
    return predicted == expected


def run_baseline_evaluation():
    """Run the OCR -> extraction pipeline over every sample and compare
    against ground truth. Returns (per_field_accuracy, overall_accuracy,
    passed_no_crash) or None if there is nothing to evaluate."""
    from app.extraction import extract_fields
    from app.ocr import run_ocr

    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())
    images = sorted(SAMPLE_DIR.glob("*.png"))
    if not images:
        print(f"❌ No sample images found in {SAMPLE_DIR}.")
        return None

    field_correct = {f: 0 for f in FIELDS}
    total_docs = 0
    crashed = 0

    for img_path in images:
        expected = ground_truth.get(img_path.name)
        if expected is None:
            print(f"  ⚠️  {img_path.name}: no ground truth entry, skipping")
            continue
        total_docs += 1
        try:
            ocr_result = run_ocr(str(img_path))
            predicted = extract_fields(ocr_result)
        except Exception as e:
            crashed += 1
            print(f"  ❌ {img_path.name}: raised {type(e).__name__}: {e}")
            continue

        row_results = []
        for field in FIELDS:
            pred_value = getattr(predicted, field)
            match = _fields_match(pred_value, expected.get(field), field)
            if match:
                field_correct[field] += 1
            row_results.append(f"{field}={'✅' if match else '❌ (' + repr(pred_value) + ' != ' + repr(expected.get(field)) + ')'}")
        print(f"  {img_path.name}: " + ", ".join(row_results))

    if total_docs == 0:
        print("❌ No samples had a matching ground truth entry to evaluate against.")
        return None

    per_field_accuracy = {
        f: field_correct[f] / total_docs for f in FIELDS
    }
    overall_accuracy = sum(field_correct.values()) / (total_docs * len(FIELDS))
    no_crash = crashed == 0

    return per_field_accuracy, overall_accuracy, no_crash, total_docs


def record_experiment(per_field_accuracy, overall_accuracy, total_docs) -> None:
    """Append this run's metrics to experiments.csv, per
    docs/03-MODEL-DEVELOPMENT.md §4 ("simple experiments.csv log")."""
    is_new = not EXPERIMENTS_LOG.exists()
    with open(EXPERIMENTS_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow([
                "date", "model", "dataset", "n_samples",
                "merchant_name_acc", "date_acc", "total_amount_acc",
                "overall_field_accuracy", "notes",
            ])
        writer.writerow([
            dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rule_based_baseline (Phase 2)",
            "synthetic_sample_receipts (SROIE stand-in)",
            total_docs,
            f"{per_field_accuracy['merchant_name']:.3f}",
            f"{per_field_accuracy['date']:.3f}",
            f"{per_field_accuracy['total_amount']:.3f}",
            f"{overall_accuracy:.3f}",
            "Baseline for Phase 3 (LayoutLMv3) to beat. Synthetic data, not real SROIE.",
        ])
    print(f"✅ Recorded metrics to {EXPERIMENTS_LOG}")


def main() -> None:
    print("🔎 Validating Phase 2 baseline structured extraction...\n")
    ok = check_ground_truth_exists()
    if not ok:
        sys.exit(1)

    result = run_baseline_evaluation()
    if result is None:
        print("\n❌ Phase 2 validation failed — could not evaluate the extractor.")
        sys.exit(1)

    per_field_accuracy, overall_accuracy, no_crash, total_docs = result

    print(f"\nField-level accuracy (n={total_docs}):")
    for field in FIELDS:
        print(f"  {field}: {per_field_accuracy[field]:.1%}")
    print(f"  overall: {overall_accuracy:.1%}")

    if not no_crash:
        print("\n❌ Extractor raised an exception on at least one sample.")
        sys.exit(1)

    record_experiment(per_field_accuracy, overall_accuracy, total_docs)

    print("\n🎉 Phase 2 exit criteria met: baseline field-level accuracy "
          "measured and recorded (this is the benchmark Phase 3 must beat).")
    print("Next: python -m uvicorn app.main:app --reload, then POST an "
          "image to http://localhost:8000/v1/extraction/baseline")


if __name__ == "__main__":
    main()
