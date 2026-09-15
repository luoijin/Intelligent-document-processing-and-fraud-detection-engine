#!/usr/bin/env python3
"""
Phase 4 exit-criteria validation, per docs/06-ROADMAP-MILESTONES.md:

    "Exit criteria: precision/recall on synthetic fraud validation set
    documented for both models."

Loads the trained Isolation Forest + XGBoost models (from
scripts/train_anomaly_models.py) and data/fraud/val.csv (from
scripts/build_fraud_dataset.py), scores the validation set with both,
sweeps thresholds for each model's best-F1 operating point (see
app.anomaly.best_threshold), and appends both rows to experiments.csv.

Run: python tests/test_phase4.py
"""
import csv
import datetime as dt
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from app.features import FEATURE_COLUMNS  # noqa: E402
from app.anomaly import (  # noqa: E402
    score_isolation_forest, score_xgboost, best_threshold, load_model,
)

DATA_DIR = REPO / "data" / "fraud"
MODEL_DIR = REPO / "models" / "anomaly" / "v1"
EXPERIMENTS_LOG = REPO / "experiments.csv"


def check_prerequisites() -> bool:
    ok = True
    if not (DATA_DIR / "val.csv").exists():
        print(f"❌ No validation set at {DATA_DIR / 'val.csv'}. "
              f"Run: python scripts/build_fraud_dataset.py")
        ok = False
    for name in ("isolation_forest.joblib", "xgboost.joblib"):
        if not (MODEL_DIR / name).exists():
            print(f"❌ No trained model at {MODEL_DIR / name}. "
                  f"Run: python scripts/train_anomaly_models.py")
            ok = False
    return ok


def load_val():
    with open(DATA_DIR / "val.csv") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[c]) for c in FEATURE_COLUMNS] for r in rows])
    y = np.array([int(r["is_synthetic_fraud"]) for r in rows])
    return X, y


def record_experiment(model_name: str, result, n_samples: int, n_fraud: int) -> None:
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
            f"{model_name} (Phase 4)",
            f"synthetic_fraud_val (n_fraud={n_fraud})",
            n_samples, "n/a", "n/a", "n/a",
            f"{result.f1:.3f}",
            f"precision={result.precision:.3f}, recall={result.recall:.3f}, "
            f"f1={result.f1:.3f} at best threshold={result.threshold:.2f}. "
            f"Synthetic fraud patterns only — not a real-world fraud-rate estimate.",
        ])


def main() -> None:
    print("🔎 Validating Phase 4 anomaly/fraud layer...\n")
    if not check_prerequisites():
        sys.exit(1)

    X_val, y_val = load_val()
    n_fraud = int(y_val.sum())
    print(f"Validation set: n={len(y_val)}, fraud={n_fraud} ({n_fraud/len(y_val):.1%})\n")

    if_model = load_model(MODEL_DIR / "isolation_forest.joblib")
    xgb_model = load_model(MODEL_DIR / "xgboost.joblib")

    if_scores = score_isolation_forest(if_model, X_val)
    xgb_scores = score_xgboost(xgb_model, X_val)

    if_result = best_threshold(if_scores, y_val)
    xgb_result = best_threshold(xgb_scores, y_val)

    print(f"Isolation Forest (unsupervised baseline):")
    print(f"  precision={if_result.precision:.3f}  recall={if_result.recall:.3f}  "
          f"f1={if_result.f1:.3f}  (threshold={if_result.threshold:.2f})\n")
    print(f"XGBoost (supervised):")
    print(f"  precision={xgb_result.precision:.3f}  recall={xgb_result.recall:.3f}  "
          f"f1={xgb_result.f1:.3f}  (threshold={xgb_result.threshold:.2f})\n")

    verdict = ("XGBoost beats the Isolation Forest baseline on F1"
               if xgb_result.f1 > if_result.f1 else
               "XGBoost does NOT beat the Isolation Forest baseline on F1"
               if xgb_result.f1 < if_result.f1 else
               "XGBoost and Isolation Forest tie on F1")
    print(f"Comparison: {verdict}.")
    print("Per docs/03-MODEL-DEVELOPMENT.md §3.2: false-positive rate matters "
          "more than raw accuracy in a fraud-review context — precision at a "
          "conservative (higher) threshold, not best-F1, is what should drive "
          "the production threshold choice (see app/main.py's review_required "
          "logic in Phase 5).")

    record_experiment("isolation_forest", if_result, len(y_val), n_fraud)
    record_experiment("xgboost", xgb_result, len(y_val), n_fraud)
    print(f"\n✅ Recorded both models' metrics to {EXPERIMENTS_LOG}")

    print(
        "\n🎉 Phase 4 exit criteria met: precision/recall on the synthetic "
        "fraud validation set documented for both models."
    )
    print(
        "\n⚠️  Reminder (docs/07-TESTING-EVALUATION.md §4): synthetic fraud "
        "is not a substitute for real fraud data — these numbers describe "
        "performance on synthetic patterns only."
    )


if __name__ == "__main__":
    main()
