"""
app.anomaly
------------
Anomaly / Fraud Model component (docs/01-ARCHITECTURE.md §2.5,
docs/03-MODEL-DEVELOPMENT.md §3.2). Consumes feature vectors produced
by app.features (stable FEATURE_COLUMNS interface) — this module does
not build features and does not extract fields, only scores.

Two models, per the documented plan:
    - Isolation Forest (unsupervised): the v1 baseline, no labels
      needed. Score: higher = more anomalous.
    - XGBoost (supervised): trained once synthetic-fraud labels exist,
      evaluated against the Isolation Forest baseline (never adopted
      without that comparison, per project instructions §5.4).

Both models expose the same `predict_fraud(X) -> (scores, labels)`
shape so app/main.py's serving code doesn't need to know which one is
loaded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier


@dataclass
class EvalResult:
    precision: float
    recall: float
    f1: float
    threshold: float


def train_isolation_forest(X_train: np.ndarray, contamination: float) -> IsolationForest:
    """contamination should reflect the expected fraud rate in the
    training population (here, the measured synthetic fraud rate from
    scripts/build_fraud_dataset.py's summary.json) — Isolation Forest
    uses it to calibrate its internal decision threshold. This is the
    one place the "unsupervised" model is told anything about labels;
    it never sees y_train itself.
    """
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X_train)
    return model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def score_isolation_forest(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Returns anomaly scores in [0, 1], higher = more anomalous
    (sklearn's raw decision_function is the opposite sign and
    unbounded, so it's inverted and squashed here for a stable,
    model-agnostic interface with XGBoost's predict_proba)."""
    raw = model.decision_function(X)  # higher = more normal
    inverted = -raw
    return (inverted - inverted.min()) / (inverted.max() - inverted.min() + 1e-9)


def score_xgboost(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def evaluate(scores: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> EvalResult:
    y_pred = (scores >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return EvalResult(precision=precision, recall=recall, f1=f1, threshold=threshold)


def best_threshold(scores: np.ndarray, y_true: np.ndarray) -> EvalResult:
    """Sweep thresholds and return the one maximizing F1 on this set.
    Used only for documenting each model's best-case operating point in
    tests/test_phase4.py — the threshold actually used in production
    (Phase 5) is a separate, deliberately conservative choice (see
    app/main.py), since false positives and false negatives don't cost
    the same in a fraud-review context (docs/03-MODEL-DEVELOPMENT.md §3.2)."""
    best = EvalResult(0, 0, 0, 0.5)
    for t in np.arange(0.05, 0.95, 0.05):
        result = evaluate(scores, y_true, threshold=float(t))
        if result.f1 > best.f1:
            best = result
    return best


def save_model(model, path: Path) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    import joblib
    return joblib.load(path)
