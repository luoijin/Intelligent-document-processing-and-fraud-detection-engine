"""
app.features
-------------
Feature Builder (docs/01-ARCHITECTURE.md §2.4): converts a structured
extracted record + historical context into the tabular feature vector
consumed by the anomaly/fraud model. This is a separate component from
both the extractor (upstream) and the anomaly model (downstream) —
per the project's one-responsibility rule, it does not extract fields
and does not score anomalies, only builds features.

Feature set per docs/03-MODEL-DEVELOPMENT.md §3.1:
    amount_zscore, is_round_number, days_since_last_from_merchant,
    duplicate_hash_flag, ocr_confidence_avg, date_plausibility

FEATURE_COLUMNS defines the stable, ordered interface to the anomaly
model — training and serving code must both go through
`build_feature_vector` so the column order can never drift between
train and inference.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Optional

FEATURE_COLUMNS = [
    "amount_zscore",
    "is_round_number",
    "days_since_last_from_merchant",
    "duplicate_hash_flag",
    "ocr_confidence_avg",
    "date_plausibility",
]

# date_plausibility thresholds: a receipt dated further in the future
# than this, or older than this, is treated as fully implausible (1.0).
# Chosen per docs/03-MODEL-DEVELOPMENT.md §3.1's description
# ("future-dated or implausibly old") — no fixed value is specified
# there, so this is a documented v1 default, not a value from the docs.
MAX_FUTURE_DAYS = 1
MAX_AGE_DAYS = 365 * 2

# Fallback used when a merchant has no prior history at all (cold
# start — e.g. this API instance has never seen this merchant before).
# 0.0 = "not anomalous by amount", since there's no history to deviate
# from. Documented default, not a data-derived value.
COLD_START_ZSCORE = 0.0


@dataclass
class MerchantStats:
    mean_amount: float
    std_amount: float
    n_observations: int


def compute_merchant_stats(records: Iterable[dict]) -> dict[str, MerchantStats]:
    """Compute per-merchant amount mean/std from a set of records.

    CRITICAL (leakage): callers must pass only clean, pre-fraud-injection
    records here (e.g. the train split before synthetic fraud is
    injected — see docs/02-DATA-STRATEGY.md §5). Computing "normal"
    merchant behavior from a set that already contains injected
    amount-inflation fraud would bake the fraud into what's considered
    normal, defeating the z-score feature's purpose.
    """
    by_merchant: dict[str, list[float]] = {}
    for r in records:
        by_merchant.setdefault(r["merchant_name"], []).append(r["total_amount"])

    stats: dict[str, MerchantStats] = {}
    for merchant, amounts in by_merchant.items():
        n = len(amounts)
        mean = sum(amounts) / n
        variance = sum((a - mean) ** 2 for a in amounts) / n if n > 1 else 0.0
        std = variance ** 0.5
        stats[merchant] = MerchantStats(mean_amount=mean, std_amount=std, n_observations=n)
    return stats


def _record_hash(record: dict) -> str:
    """Fuzzy-ish duplicate key: merchant + date + total, rounded to the
    cent. Two records with identical (merchant, date, amount) are
    treated as a duplicate submission per docs/02-DATA-STRATEGY.md §4."""
    key = f"{record['merchant_name']}|{record['date']}|{round(record['total_amount'], 2)}"
    return hashlib.sha256(key.encode()).hexdigest()


def _is_round_number(amount: float, threshold: float = 100.0) -> bool:
    """docs/03-MODEL-DEVELOPMENT.md §3.1: 'total ends in .00 and > threshold'."""
    return amount > threshold and round(amount, 2) == round(amount)


def _date_plausibility(record_date: date, reference_date: Optional[date] = None) -> float:
    """0.0 = fully plausible, 1.0 = fully implausible. Linear ramp
    between the "just barely fine" and "clearly wrong" thresholds
    rather than a hard 0/1 cutoff, so the feature carries some signal
    strength rather than just being another boolean flag."""
    reference_date = reference_date or date.today()
    delta_days = (record_date - reference_date).days

    if delta_days > MAX_FUTURE_DAYS:
        return 1.0
    if delta_days < -MAX_AGE_DAYS:
        return 1.0
    if delta_days >= 0:
        # future-dated but within MAX_FUTURE_DAYS tolerance
        return delta_days / max(MAX_FUTURE_DAYS, 1)
    # past-dated: ramp from 0 (today) to 1 (MAX_AGE_DAYS ago)
    age_days = -delta_days
    return min(age_days / MAX_AGE_DAYS, 1.0)


def build_feature_vector(
    record: dict,
    merchant_stats: dict[str, MerchantStats],
    prior_hashes: set[str],
    reference_date: Optional[date] = None,
) -> dict:
    """Build the feature vector for one structured record.

    `record` shape (subset of docs/02-DATA-STRATEGY.md §2 target schema):
        {"merchant_name": str, "date": "YYYY-MM-DD", "total_amount": float,
         "ocr_confidence_avg": float (0-100),
         "days_since_last_from_merchant": float | None (precomputed by
             caller from the record sequence; this module doesn't own
             sequencing/ordering, only the per-record math)}

    `prior_hashes` is the set of `_record_hash` values seen so far
    (in whatever scope the caller defines — e.g. all training records
    processed before this one) — used for `duplicate_hash_flag`.
    Callers own the scope/ordering of what counts as "prior"; this
    function only checks membership.
    """
    merchant = record["merchant_name"]
    amount = record["total_amount"]
    record_date = record["date"]
    if isinstance(record_date, str):
        record_date = datetime.strptime(record_date, "%Y-%m-%d").date()

    stats = merchant_stats.get(merchant)
    if stats is None or stats.std_amount == 0:
        amount_zscore = COLD_START_ZSCORE
    else:
        amount_zscore = (amount - stats.mean_amount) / stats.std_amount

    record_hash = _record_hash(record)
    duplicate_hash_flag = 1 if record_hash in prior_hashes else 0

    days_since_last = record.get("days_since_last_from_merchant")
    if days_since_last is None:
        days_since_last = -1  # sentinel: no prior record for this merchant

    return {
        "amount_zscore": amount_zscore,
        "is_round_number": int(_is_round_number(amount)),
        "days_since_last_from_merchant": days_since_last,
        "duplicate_hash_flag": duplicate_hash_flag,
        "ocr_confidence_avg": record.get("ocr_confidence_avg", 100.0),
        "date_plausibility": _date_plausibility(record_date, reference_date),
    }


def to_vector(features: dict) -> list[float]:
    """Project a feature dict onto the stable FEATURE_COLUMNS order."""
    return [float(features[c]) for c in FEATURE_COLUMNS]
