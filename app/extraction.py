"""
app.extraction
---------------
Rule-based structured field extractor for the IDP pipeline (Phase 2).

Per docs/03-MODEL-DEVELOPMENT.md §2.1, this is the "baseline (build
first)" entity extraction layer: regex / positional heuristics over
OCR tokens, with no training cost, that produces a number to beat once
Phase 3 introduces a fine-tuned LayoutLMv3 model.

Heuristics implemented (per §2.1 and docs/01-ARCHITECTURE.md §2.3):
    - `total_amount`: largest currency-formatted number on a line that
      also contains a total/amount-due keyword (falls back to the
      largest currency-formatted number anywhere on the document if no
      such line is found).
    - `date`: first token matching a common date pattern.
    - `merchant_name`: the topmost text line (receipts conventionally
      lead with the merchant header).

Input is the token list produced by `app.ocr.run_ocr` (see
`app.ocr.OcrToken`) so this module has no dependency on the OCR engine
choice itself — only on the stable `OcrToken` shape.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from app.ocr import OcrResult, OcrToken

# ---------------------------------------------------------------------------
# Heuristic constants
# ---------------------------------------------------------------------------

# Keywords that mark a line as a plausible "total" line. Checked as a
# whole-word match (case-insensitive) against the line's joined text.
TOTAL_KEYWORDS = ("total", "amount due", "balance due", "amount")

# A currency-formatted number: optional thousands separators, optional
# 1-2 decimal places. Deliberately does not require a currency symbol
# since none of the current sample data includes one (see
# `docs/CHANGELOG/CHANGELOG_PHASE2.md` "Known limitations").
_CURRENCY_NUMBER_RE = re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?$")

# (regex, strptime format) pairs, tried in order. Each regex is matched
# against a single OCR token (or a small joined window of tokens, see
# `_candidate_date_strings`) rather than the full raw text, so a match's
# span unambiguously identifies which token(s) to report.
_DATE_PATTERNS: List[tuple[re.Pattern, str]] = [
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "%Y-%m-%d"),          # 2026-09-01
    (re.compile(r"^\d{4}/\d{2}/\d{2}$"), "%Y/%m/%d"),          # 2026/09/01
    (re.compile(r"^\d{2}/\d{2}/\d{4}$"), "%m/%d/%Y"),          # 09/01/2026
    (re.compile(r"^\d{2}-\d{2}-\d{4}$"), "%m-%d-%Y"),          # 09-01-2026
    (re.compile(r"^\d{2}\.\d{2}\.\d{4}$"), "%d.%m.%Y"),        # 01.09.2026
    (re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{4}$"), "%d-%b-%Y"),  # 01-Sep-2026
    (re.compile(r"^\d{1,2}/[A-Za-z]{3}/\d{4}$"), "%d/%b/%Y"),  # 01/Sep/2026
]


@dataclass
class ExtractedFields:
    merchant_name: Optional[str] = None
    date: Optional[str] = None  # normalized to YYYY-MM-DD when parseable
    total_amount: Optional[float] = None
    currency: Optional[str] = None
    extraction_confidence: float = 0.0


# ---------------------------------------------------------------------------
# Line grouping — OCR tokens arrive as a flat list; the heuristics below
# all reason in terms of receipt *lines*, so tokens are first grouped by
# vertical (top) proximity and sorted left-to-right within each line.
# ---------------------------------------------------------------------------

def _group_lines(tokens: List[OcrToken]) -> List[List[OcrToken]]:
    """Group tokens into visual lines by proximity of their `top`
    coordinate, then sort each line's tokens left-to-right.

    Tolerance is half of the current line's first token height, which
    is robust to normal per-character height variance within one line
    of text without merging genuinely separate lines.
    """
    if not tokens:
        return []

    ordered = sorted(tokens, key=lambda t: t.top)
    lines: List[List[OcrToken]] = []
    current: List[OcrToken] = [ordered[0]]
    current_top = ordered[0].top
    tolerance = max(5, ordered[0].height // 2)

    for token in ordered[1:]:
        if abs(token.top - current_top) <= tolerance:
            current.append(token)
        else:
            lines.append(current)
            current = [token]
            current_top = token.top
            tolerance = max(5, token.height // 2)
    lines.append(current)

    return [sorted(line, key=lambda t: t.left) for line in lines]


def _line_text(line: List[OcrToken]) -> str:
    return " ".join(t.text for t in line)


# ---------------------------------------------------------------------------
# Field heuristics
# ---------------------------------------------------------------------------

def _parse_currency_number(text: str) -> Optional[float]:
    """Parse a token's text as a currency-formatted number, stripped of
    a small set of common currency symbols/prefixes. Returns None if
    the (symbol-stripped) text doesn't match the currency number
    pattern — callers should not assume every numeric-looking token is
    a monetary amount (e.g. a date fragment or item count)."""
    stripped = text.strip().lstrip("$₱").rstrip(":")
    if stripped.upper().startswith("PHP"):
        stripped = stripped[3:]
    if not _CURRENCY_NUMBER_RE.match(stripped):
        return None
    try:
        return float(stripped.replace(",", ""))
    except ValueError:
        return None


def _extract_total_amount(lines: List[List[OcrToken]]) -> Optional[float]:
    def amounts_in(candidate_lines: List[List[OcrToken]]) -> List[float]:
        values: List[float] = []
        for line in candidate_lines:
            for token in line:
                amount = _parse_currency_number(token.text)
                if amount is not None:
                    values.append(amount)
        return values

    keyword_lines = [
        line for line in lines
        if any(kw in _line_text(line).lower() for kw in TOTAL_KEYWORDS)
    ]
    candidates = amounts_in(keyword_lines)
    if not candidates:
        # Fallback per docs/03-MODEL-DEVELOPMENT.md §2.1: "largest
        # currency-formatted number" anywhere, when no keyword line
        # is found (e.g. OCR misread "TOTAL" itself).
        candidates = amounts_in(lines)

    return max(candidates) if candidates else None


def _extract_date(lines: List[List[OcrToken]]) -> Optional[str]:
    for line in lines:
        for token in line:
            candidate = token.text.strip().rstrip(",")
            for pattern, fmt in _DATE_PATTERNS:
                if pattern.match(candidate):
                    try:
                        parsed = datetime.strptime(candidate, fmt)
                    except ValueError:
                        continue
                    return parsed.strftime("%Y-%m-%d")
    return None


def _extract_merchant_name(lines: List[List[OcrToken]]) -> Optional[str]:
    """Baseline heuristic: the topmost non-empty line. Real receipts
    conventionally lead with the merchant name/header; this is a
    positional proxy for "largest/topmost text block" per
    `docs/03-MODEL-DEVELOPMENT.md` §2.1 — Tesseract's `image_to_data`
    does not expose font size, so topmost-line position is used as the
    available signal rather than true visual size. See
    `docs/CHANGELOG/CHANGELOG_PHASE2.md` for this documented deviation."""
    for line in lines:
        text = _line_text(line).strip()
        if text:
            return text
    return None


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def extract_fields(ocr_result: OcrResult) -> ExtractedFields:
    """Run the Phase 2 baseline rule-based extractor over an OCR result.

    Currency is intentionally left unset (`None`) — none of the
    heuristics above detect a currency symbol/code, and none of the
    current sample data includes one to detect. This is a documented
    known limitation (see CHANGELOG_PHASE2.md), not a silent gap.
    """
    lines = _group_lines(ocr_result.tokens)

    merchant_name = _extract_merchant_name(lines)
    date = _extract_date(lines)
    total_amount = _extract_total_amount(lines)

    fields_found = sum(f is not None for f in (merchant_name, date, total_amount))
    extraction_confidence = fields_found / 3.0

    return ExtractedFields(
        merchant_name=merchant_name,
        date=date,
        total_amount=total_amount,
        currency=None,
        extraction_confidence=extraction_confidence,
    )


def to_dict(fields: ExtractedFields) -> dict:
    """Serialize an ExtractedFields to a JSON-compatible dict (for API
    responses), matching the relevant subset of the target schema in
    docs/02-DATA-STRATEGY.md §2 (document_id/line_items are out of
    scope until Phase 5's persistence layer exists)."""
    return {
        "merchant_name": fields.merchant_name,
        "date": fields.date,
        "total_amount": fields.total_amount,
        "currency": fields.currency,
        "extraction_confidence": round(fields.extraction_confidence, 2),
    }
