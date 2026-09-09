"""
scripts.generate_sample_receipts
---------------------------------
Generates a small set of synthetic receipt images so the Phase 1/2
pipeline can be exercised end-to-end without a network dependency.

IMPORTANT: This is a placeholder data source, NOT the SROIE dataset
referenced in docs/02-DATA-STRATEGY.md §1. SROIE requires a manual
download the user performs themselves (research-use dataset, not
fetchable from this environment's allowed network egress). Once
downloaded, drop SROIE images into data/raw/ and the same
app.ocr.run_ocr() pipeline runs against them unchanged.

Phase 2 addition: alongside each image, this script now also writes
`ground_truth.json` (merchant_name/date/total_amount per sample) so
`tests/test_phase2.py` has something to measure the rule-based
extractor's field-level accuracy against — the same synthetic-stand-in
role this script already played for Phase 1's OCR exit criteria. This
is NOT SROIE's real ground-truth annotations; see
`docs/CHANGELOG/CHANGELOG_PHASE2.md` for the same class of deviation
already documented for Phase 1's OCR validation.

Run: python scripts/generate_sample_receipts.py
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sample_receipts"

SAMPLE_RECEIPTS = [
    {
        "lines": [
            "GROCERY MART",
            "123 Main St, Cebu City",
            "Date: 2026-09-01",
            "",
            "Milk 1L         89.00",
            "Bread           65.50",
            "Eggs (12pc)    145.00",
            "",
            "TOTAL          299.50",
            "Thank you!",
        ],
        "ground_truth": {
            "merchant_name": "GROCERY MART",
            "date": "2026-09-01",
            "total_amount": 299.50,
        },
    },
    {
        "lines": [
            "CAFE BREW HOUSE",
            "45 Coffee Ave",
            "Date: 2026-08-15",
            "",
            "Latte           120.00",
            "Croissant        85.00",
            "",
            "TOTAL           205.00",
            "Visit again soon",
        ],
        "ground_truth": {
            "merchant_name": "CAFE BREW HOUSE",
            "date": "2026-08-15",
            "total_amount": 205.00,
        },
    },
    {
        "lines": [
            "HARDWARE PLUS",
            "78 Builder Rd",
            "Date: 2026-07-22",
            "",
            "Screws (box)     45.00",
            "Hammer          350.00",
            "Paint (1L)      220.00",
            "",
            "TOTAL           615.00",
        ],
        "ground_truth": {
            "merchant_name": "HARDWARE PLUS",
            "date": "2026-07-22",
            "total_amount": 615.00,
        },
    },
]


def render_receipt(lines: list[str], skew_degrees: float = 0.0) -> Image.Image:
    width, height = 400, 500
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        # Older Pillow versions don't accept a size kwarg on load_default.
        font = ImageFont.load_default()

    y = 20
    for line in lines:
        draw.text((20, y), line, fill=0, font=font)
        y += 32

    if skew_degrees:
        img = img.rotate(skew_degrees, expand=True, fillcolor=255)

    return img


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    skews = [0.0, 2.5, -3.0]
    ground_truth: dict[str, dict] = {}

    for i, (sample, skew) in enumerate(zip(SAMPLE_RECEIPTS, skews), start=1):
        img = render_receipt(sample["lines"], skew_degrees=skew)
        filename = f"sample_receipt_{i:02d}.png"
        out_path = OUTPUT_DIR / filename
        img.save(out_path)
        ground_truth[filename] = sample["ground_truth"]
        print(f"Wrote {out_path} (skew={skew}deg)")

    gt_path = OUTPUT_DIR / "ground_truth.json"
    gt_path.write_text(json.dumps(ground_truth, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {gt_path} ({len(ground_truth)} entries)")


if __name__ == "__main__":
    main()
