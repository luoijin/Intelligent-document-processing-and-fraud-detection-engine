"""
scripts.build_fraud_dataset
-----------------------------
Phase 4 (docs/03-MODEL-DEVELOPMENT.md §3, docs/02-DATA-STRATEGY.md §4-5)
data prep:

    1. Generate a clean synthetic corpus of structured extraction
       records (this stands in for "records that came out of Phase
       2/3's extractor" — no images/OCR involved here, this is the
       Feature Builder's input shape, per docs/01-ARCHITECTURE.md
       stage E onward, which is deliberately image-agnostic).
    2. Split 70/15/15 train/val/test BEFORE any fraud injection, per
       docs/02-DATA-STRATEGY.md §5 ("split before injection to avoid
       leakage").
    3. Inject synthetic fraud into each split independently (so a
       fraud pattern generated from a train record can never leak into
       val/test), using the 5 methods in docs/02-DATA-STRATEGY.md §4.
    4. Compute merchant statistics from ONLY the clean (pre-injection)
       train records — see app/features.py's leakage warning — and use
       those same frozen stats for feature-building on train/val/test
       alike, exactly like a real deployment would (stats learned once
       from history, applied to new incoming records).
    5. Build feature vectors via app/features.py (the same code path
       Phase 5's live API uses) and write data/fraud/{split}.csv.

Run: python scripts/build_fraud_dataset.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from app.features import (  # noqa: E402
    FEATURE_COLUMNS,
    build_feature_vector,
    compute_merchant_stats,
)

OUTPUT_DIR = REPO / "data" / "fraud"
SEED = 42
N_CLEAN_RECORDS = 900
FRAUD_RATE = 0.08  # 8% of each split becomes synthetic fraud

MERCHANTS = {
    "GROCERY MART": 250.0,
    "CAFE BREW HOUSE": 180.0,
    "CITY PHARMACY": 320.0,
    "QUICKSTOP CONVENIENCE": 90.0,
    "THE STEAKHOUSE": 1200.0,
    "URBAN HARDWARE": 450.0,
    "BOOKNOOK": 150.0,
    "FRESH LAUNDRY": 60.0,
    "TECH REPAIR SHOP": 800.0,
    "GREENLEAF DINER": 220.0,
}
REFERENCE_DATE = date(2026, 9, 1)


def generate_clean_records(rng: random.Random) -> list[dict]:
    """Generate N_CLEAN_RECORDS plausible, non-fraudulent transactions
    spread across MERCHANTS with per-merchant amount distributions and
    dates within the last year, sorted chronologically per merchant so
    days_since_last_from_merchant can be computed sequentially."""
    records = []
    for i in range(N_CLEAN_RECORDS):
        merchant = rng.choice(list(MERCHANTS.keys()))
        base_amount = MERCHANTS[merchant]
        amount = max(5.0, round(rng.gauss(base_amount, base_amount * 0.18), 2))
        days_ago = rng.randint(1, 365)
        record_date = REFERENCE_DATE - timedelta(days=days_ago)
        ocr_conf = round(max(40.0, min(99.9, rng.gauss(92.0, 5.0))), 1)
        records.append({
            "document_id": f"clean-{i:04d}",
            "merchant_name": merchant,
            "date": record_date.isoformat(),
            "total_amount": amount,
            "ocr_confidence_avg": ocr_conf,
            "is_synthetic_fraud": False,
        })

    records.sort(key=lambda r: (r["merchant_name"], r["date"]))
    last_seen: dict[str, date] = {}
    for r in records:
        d = date.fromisoformat(r["date"])
        prev = last_seen.get(r["merchant_name"])
        r["days_since_last_from_merchant"] = (d - prev).days if prev else None
        last_seen[r["merchant_name"]] = d
    return records


def split_records(records: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    shuffled = records[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def inject_fraud(split_records_: list[dict], rng: random.Random) -> list[dict]:
    """Injects fraud per docs/02-DATA-STRATEGY.md §4, independently
    within this split only (never reaches across splits)."""
    out = [dict(r) for r in split_records_]
    n_fraud = int(len(out) * FRAUD_RATE)
    fraud_indices = rng.sample(range(len(out)), min(n_fraud, len(out)))
    methods = [
        "amount_inflation", "duplicate_submission",
        "date_manipulation", "merchant_mismatch", "round_number_stuffing",
    ]

    injected = []
    for i, idx in enumerate(fraud_indices):
        method = methods[i % len(methods)]
        record = dict(out[idx])
        record["is_synthetic_fraud"] = True
        record["fraud_method"] = method

        if method == "amount_inflation":
            record["total_amount"] = round(record["total_amount"] * rng.uniform(5, 20), 2)
        elif method == "duplicate_submission":
            dup = dict(record)
            dup["document_id"] = record["document_id"] + "-dup"
            injected.append(dup)
        elif method == "date_manipulation":
            shift = rng.choice([rng.randint(30, 90), -rng.randint(800, 1200)])
            record["date"] = (date.fromisoformat(record["date"]) + timedelta(days=shift)).isoformat()
        elif method == "merchant_mismatch":
            other = rng.choice([m for m in MERCHANTS if m != record["merchant_name"]])
            record["merchant_name"] = other
        elif method == "round_number_stuffing":
            record["total_amount"] = float(rng.choice([500, 1000, 1500, 2000]))

        out[idx] = record

    out.extend(injected)
    return out


def build_features_for_split(
    records: list[dict], merchant_stats, prior_hashes: set
) -> tuple[list[list[float]], list[int]]:
    X, y = [], []
    for r in records:
        features = build_feature_vector(r, merchant_stats, prior_hashes, reference_date=REFERENCE_DATE)
        X.append([features[c] for c in FEATURE_COLUMNS])
        y.append(int(r["is_synthetic_fraud"]))
    return X, y


def write_split_csv(path: Path, records: list[dict], X: list[list[float]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["document_id", "merchant_name", "date", "total_amount",
                          *FEATURE_COLUMNS, "is_synthetic_fraud", "fraud_method"])
        for r, x in zip(records, X):
            writer.writerow([
                r["document_id"], r["merchant_name"], r["date"], r["total_amount"],
                *x, int(r["is_synthetic_fraud"]), r.get("fraud_method", ""),
            ])


def main() -> None:
    rng = random.Random(SEED)

    clean = generate_clean_records(rng)
    splits = split_records(clean, rng)

    # Merchant stats: computed ONCE from the clean, pre-injection TRAIN
    # split only, then frozen and reused for val/test feature-building
    # (see app/features.py docstring — this is the leakage guard).
    merchant_stats = compute_merchant_stats(splits["train"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, split in splits.items():
        fraud_split = inject_fraud(split, rng)
        # duplicate-hash detection scope: "prior" = every other record
        # in this same split (order-independent for this synthetic
        # eval; a live system would instead use "everything persisted
        # so far" — see app/store.py).
        from app.features import _record_hash
        all_hashes = [_record_hash(r) for r in fraud_split]
        prior_hashes_per_record = []
        seen: set[str] = set()
        for h in all_hashes:
            prior_hashes_per_record.append(set(seen))
            seen.add(h)

        X, y = [], []
        for r, prior in zip(fraud_split, prior_hashes_per_record):
            features = build_feature_vector(r, merchant_stats, prior, reference_date=REFERENCE_DATE)
            X.append([features[c] for c in FEATURE_COLUMNS])
            y.append(int(r["is_synthetic_fraud"]))

        write_split_csv(OUTPUT_DIR / f"{name}.csv", fraud_split, X)
        summary[name] = {"n": len(fraud_split), "n_fraud": sum(y),
                          "fraud_rate": round(sum(y) / len(y), 4)}
        print(f"{name}: n={len(fraud_split)}, fraud={sum(y)} "
              f"({summary[name]['fraud_rate']:.1%})")

    (OUTPUT_DIR / "merchant_stats.json").write_text(json.dumps(
        {m: vars(s) for m, s in merchant_stats.items()}, indent=2))
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote train/val/test CSVs + merchant_stats.json to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
