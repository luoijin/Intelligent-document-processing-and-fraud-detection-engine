"""
app.store
----------
Result Store component (docs/01-ARCHITECTURE.md §2.7): SQLite
persistence for processed documents. v1 default per
docs/05-DEPLOYMENT-FREE-TIER.md ("SQLite locally"). This module owns
persistence only — it does not extract fields, build features, or
score anomalies; `app/main.py` wires those components together and
calls this one to read/write records.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

DB_PATH = Path(__file__).parent.parent / "data" / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    merchant_name TEXT,
    date TEXT,
    total_amount REAL,
    currency TEXT,
    extraction_confidence REAL,
    ocr_confidence_avg REAL,
    anomaly_score REAL,
    review_required INTEGER,
    record_hash TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_merchant ON documents(merchant_name);
"""


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def insert_document(record: dict) -> None:
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO documents
               (document_id, merchant_name, date, total_amount, currency,
                extraction_confidence, ocr_confidence_avg, anomaly_score,
                review_required, record_hash, created_at)
               VALUES (:document_id, :merchant_name, :date, :total_amount,
                       :currency, :extraction_confidence, :ocr_confidence_avg,
                       :anomaly_score, :review_required, :record_hash, :created_at)""",
            record,
        )


def get_document(document_id: str) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return dict(row) if row else None


def get_prior_records_for_merchant(merchant_name: str) -> list[dict]:
    """All previously persisted records for a merchant, used by the
    Feature Builder to compute live amount_zscore /
    days_since_last_from_merchant at request time (see app/main.py).
    Ordered oldest-first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE merchant_name = ? ORDER BY date ASC",
            (merchant_name,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_record_hashes() -> set[str]:
    """Duplicate-detection scope: every record ever persisted, per
    app/features.py's duplicate_hash_flag."""
    with get_connection() as conn:
        rows = conn.execute("SELECT record_hash FROM documents").fetchall()
        return {r["record_hash"] for r in rows if r["record_hash"]}
