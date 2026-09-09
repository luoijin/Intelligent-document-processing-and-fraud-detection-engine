"""
app.ocr
-------
OCR engine wrapper for the IDP pipeline (Phase 1).

Per docs/03-MODEL-DEVELOPMENT.md §1, Tesseract is used first ("fastest
to integrate; use first") with EasyOCR/TrOCR as later-phase upgrades.
This module wraps pytesseract behind a stable interface
(`run_ocr(image_path) -> OcrResult`) so the OCR engine can be swapped
in a later phase (Phase 3+) without changing callers.

Output shape matches docs/01-ARCHITECTURE.md §2.2: raw text tokens with
bounding boxes and per-token confidence, ready for the entity extractor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pytesseract
from pytesseract import Output

from app.preprocessing import preprocess


@dataclass
class OcrToken:
    text: str
    confidence: float  # 0-100, Tesseract's native scale
    left: int
    top: int
    width: int
    height: int


@dataclass
class OcrResult:
    tokens: List[OcrToken] = field(default_factory=list)
    raw_text: str = ""
    avg_confidence: float = 0.0


def run_ocr(image_path: str) -> OcrResult:
    """Run the full OCR step: preprocess the image, then extract text
    tokens with bounding boxes and confidence scores.

    Raises:
        ValueError: if the image cannot be decoded (propagated from
            app.preprocessing.load_image).
    """
    processed = preprocess(image_path)

    data = pytesseract.image_to_data(processed, output_type=Output.DICT)

    tokens: List[OcrToken] = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        conf_raw = data["conf"][i]
        try:
            confidence = float(conf_raw)
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence < 0:
            # Tesseract emits -1 for non-text layout regions (lines,
            # blocks); these carry no token text so this branch is a
            # defensive skip, not expected to trigger given the
            # `if not text: continue` guard above.
            continue

        tokens.append(
            OcrToken(
                text=text,
                confidence=confidence,
                left=int(data["left"][i]),
                top=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
            )
        )

    raw_text = " ".join(t.text for t in tokens)
    avg_confidence = (
        sum(t.confidence for t in tokens) / len(tokens) if tokens else 0.0
    )

    return OcrResult(tokens=tokens, raw_text=raw_text, avg_confidence=avg_confidence)


def to_dict(result: OcrResult) -> dict:
    """Serialize an OcrResult to a JSON-compatible dict (for API responses)."""
    return {
        "raw_text": result.raw_text,
        "avg_confidence": round(result.avg_confidence, 2),
        "tokens": [
            {
                "text": t.text,
                "confidence": t.confidence,
                "bbox": {"left": t.left, "top": t.top, "width": t.width, "height": t.height},
            }
            for t in result.tokens
        ],
    }
