"""
scripts.train_anomaly_models
------------------------------
Phase 4 (docs/03-MODEL-DEVELOPMENT.md §3.2) training: fits Isolation
Forest (unsupervised baseline) and XGBoost (supervised) on
data/fraud/train.csv (from scripts/build_fraud_dataset.py), evaluates
both on data/fraud/val.csv, and saves both models. Precision/recall
comparison is the Phase 4 exit-criteria metric, computed and recorded
by tests/test_phase4.py — this script only trains + saves.

Run: python scripts/train_anomaly_models.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from app.features import FEATURE_COLUMNS  # noqa: E402
from app.anomaly import (  # noqa: E402
    train_isolation_forest, train_xgboost, save_model,
)

DATA_DIR = REPO / "data" / "fraud"
MODEL_DIR = REPO / "models" / "anomaly" / "v1"


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA_DIR / f"{name}.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[c]) for c in FEATURE_COLUMNS] for r in rows])
    y = np.array([int(r["is_synthetic_fraud"]) for r in rows])
    return X, y


def main() -> None:
    if not (DATA_DIR / "train.csv").exists():
        print(f"No dataset at {DATA_DIR}. Run: python scripts/build_fraud_dataset.py")
        sys.exit(1)

    X_train, y_train = load_split("train")
    summary = json.loads((DATA_DIR / "summary.json").read_text())
    contamination = summary["train"]["fraud_rate"]

    print(f"Training on {len(X_train)} records "
          f"({int(y_train.sum())} fraud, contamination={contamination:.3f})")

    if_model = train_isolation_forest(X_train, contamination=contamination)
    xgb_model = train_xgboost(X_train, y_train)

    save_model(if_model, MODEL_DIR / "isolation_forest.joblib")
    save_model(xgb_model, MODEL_DIR / "xgboost.joblib")

    print(f"Saved models to {MODEL_DIR}")
    print("Next: python tests/test_phase4.py "
          "(evaluates both on data/fraud/val.csv, records precision/recall)")


if __name__ == "__main__":
    main()
