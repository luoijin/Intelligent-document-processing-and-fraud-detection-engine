"""
scripts.prepare_layoutlm_labels
--------------------------------
Phase 3 (docs/03-MODEL-DEVELOPMENT.md §2.2) data prep: converts each
sample's OCR tokens + Phase 2's `ground_truth.json` field values into
BIO token-classification labels for LayoutLMv3 fine-tuning.

Label set (per §2.2): O, B-MERCHANT, I-MERCHANT, B-DATE, I-DATE,
B-TOTAL, I-TOTAL.

How labels are derived
-----------------------
This is a *silver-labeling* step, not manual annotation: for each
sample, the known-correct field string (from ground_truth.json) is
matched against the OCR token stream, and matching tokens are tagged
B-/I-<FIELD>. Everything else is O. This works because the synthetic
samples are clean renders where the ground-truth string appears
verbatim in the OCR output — it will NOT work unmodified on noisy
real-world receipts, where OCR misreads mean the ground-truth string
often won't appear as an exact token match. Real SROIE/self-collected
data needs actual manual labeling per docs/02-DATA-STRATEGY.md §3
(spreadsheet or Label Studio), not this auto-alignment shortcut.

Output: one JSON-lines file, `data/layoutlm/train.jsonl`, one record
per sample:
    {
        "image": "sample_receipt_01.png",
        "tokens": ["GROCERY", "MART", ...],
        "bboxes": [[l, t, l+w, t+h], ...],   # unnormalized pixel boxes
        "labels": ["B-MERCHANT", "I-MERCHANT", ...]
    }

This intentionally does NOT normalize bboxes to LayoutLMv3's 0-1000
scale or run the HF processor here — that's done in
`train_layoutlmv3.py` at train time, alongside the image itself, since
the processor needs both.

Run: python scripts/prepare_layoutlm_labels.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from app.ocr import run_ocr  # noqa: E402
from app.extraction import _group_lines  # noqa: E402

SAMPLE_DIR = REPO / "data" / "sample_receipts"
GROUND_TRUTH_PATH = SAMPLE_DIR / "ground_truth.json"
OUTPUT_DIR = REPO / "data" / "layoutlm"
OUTPUT_PATH = OUTPUT_DIR / "train.jsonl"

FIELD_TO_LABEL = {
    "merchant_name": "MERCHANT",
    "date": "DATE",
    "total_amount": "TOTAL",
}


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _label_tokens(tokens, ground_truth: dict) -> list[str]:
    """Greedy exact-match silver labeling: for each field, look for a
    contiguous run of tokens whose concatenated normalized text equals
    the normalized ground-truth string. First match wins per field.
    """
    labels = ["O"] * len(tokens)
    norm_tokens = [_normalize(t.text) for t in tokens]

    for field, tag in FIELD_TO_LABEL.items():
        value = ground_truth.get(field)
        if value is None:
            continue
        # total_amount is a float in ground_truth.json (e.g. 299.5) but
        # renders on the receipt with a fixed 2 decimals (e.g. "299.50");
        # str(299.5) normalizes to a different digit string than the
        # rendered token, so format floats the same way the OCR text
        # actually shows them before normalizing.
        display_value = f"{value:.2f}" if isinstance(value, float) else value
        target = _normalize(str(display_value))
        if not target:
            continue

        matched = False
        for start in range(len(tokens)):
            if labels[start] != "O":
                continue
            acc = ""
            for end in range(start, len(tokens)):
                if labels[end] != "O":
                    break
                acc += norm_tokens[end]
                if acc == target:
                    labels[start] = f"B-{tag}"
                    for i in range(start + 1, end + 1):
                        labels[i] = f"I-{tag}"
                    matched = True
                    break
                if len(acc) > len(target):
                    break
            if matched:
                break

    return labels


def main() -> None:
    if not GROUND_TRUTH_PATH.exists():
        print(f"No ground truth at {GROUND_TRUTH_PATH}. "
              f"Run scripts/generate_sample_receipts.py first.")
        sys.exit(1)

    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())
    images = sorted(SAMPLE_DIR.glob("*.png"))
    if not images:
        print(f"No sample images in {SAMPLE_DIR}.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    unmatched_fields_total = 0

    for img_path in images:
        gt = ground_truth.get(img_path.name)
        if gt is None:
            print(f"  skip {img_path.name}: no ground truth entry")
            continue

        ocr_result = run_ocr(str(img_path))
        lines = _group_lines(ocr_result.tokens)
        ordered_tokens = [t for line in lines for t in line]

        labels = _label_tokens(ordered_tokens, gt)
        found_tags = {l.split("-")[1] for l in labels if l != "O"}
        expected_tags = {FIELD_TO_LABEL[f] for f, v in gt.items()
                          if v is not None and f in FIELD_TO_LABEL}
        missing = expected_tags - found_tags
        if missing:
            unmatched_fields_total += len(missing)
            print(f"  ⚠️  {img_path.name}: could not silver-label {missing} "
                  f"(ground-truth string not found verbatim in OCR tokens)")

        records.append({
            "image": img_path.name,
            "tokens": [t.text for t in ordered_tokens],
            "bboxes": [[t.left, t.top, t.left + t.width, t.top + t.height]
                       for t in ordered_tokens],
            "labels": labels,
        })
        print(f"  ✅ {img_path.name}: {len(ordered_tokens)} tokens, "
              f"labels={[l for l in labels if l != 'O']}")

    with open(OUTPUT_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(records)} labeled records to {OUTPUT_PATH}")
    if unmatched_fields_total:
        print(f"⚠️  {unmatched_fields_total} field(s) across the set could not "
              f"be silver-labeled — see warnings above.")
    print(
        "\nNOTE: this is silver-labeled synthetic data (n={}), far below the "
        "100-300 manually labeled documents docs/02-DATA-STRATEGY.md §3 "
        "calls a usable minimum for real fine-tuning. Fine-tuning on this "
        "set alone is a pipeline smoke test, not a meaningful Phase 3 "
        "result — see docs/CHANGELOG/CHANGELOG_PHASE3.md.".format(len(records))
    )


if __name__ == "__main__":
    main()
