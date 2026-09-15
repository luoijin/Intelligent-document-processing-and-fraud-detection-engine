"""
scripts.train_layoutlmv3
-------------------------
Phase 3 (docs/03-MODEL-DEVELOPMENT.md §2.2) fine-tuning script:
LayoutLMv3-base token classification over
`data/layoutlm/train.jsonl` (produced by `prepare_layoutlm_labels.py`).

WHERE TO RUN THIS: Google Colab free tier (T4 GPU), per
docs/03-MODEL-DEVELOPMENT.md §5 ("LayoutLM fine-tuning: Google Colab
free T4 GPU"). This script is NOT run as part of the automated test
suite and is NOT expected to work in the project's CPU-only local/CI
environment — it needs a GPU and network access to huggingface.co to
pull the pretrained `microsoft/layoutlmv3-base` weights, neither of
which the $0/local-first guardrail-compliant CI environment provides
on demand. This is a documented exception to "local-first": *training*
happens on free external compute (Colab), the *served* model is a
static artifact loaded locally at inference time — same pattern the
plan already uses for OCR/anomaly models, just with an extra offline
step.

Colab usage:
    1. Upload this repo (or just app/, data/layoutlm/, this script).
    2. !pip install transformers datasets seqeval accelerate
    3. !python train_layoutlmv3.py
    4. Download models/layoutlmv3/v1/ back into the repo (gitignored;
       see docs/03-MODEL-DEVELOPMENT.md §4 "never committed to git if
       large").

Exit criteria this feeds (docs/06-ROADMAP-MILESTONES.md, Phase 3):
"fine-tuned model's F1 documented and compared against the Phase 2
baseline" — this script trains and saves the model; run
`tests/test_phase3.py` afterward to compute and record that F1 against
the Phase 2 number in `experiments.csv`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
DATA_PATH = REPO / "data" / "layoutlm" / "train.jsonl"
MODEL_OUT_DIR = REPO / "models" / "layoutlmv3" / "v1"
BASE_MODEL = "microsoft/layoutlmv3-base"

LABELS = [
    "O",
    "B-MERCHANT", "I-MERCHANT",
    "B-DATE", "I-DATE",
    "B-TOTAL", "I-TOTAL",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}


def load_records() -> list[dict]:
    if not DATA_PATH.exists():
        print(f"No labeled data at {DATA_PATH}. "
              f"Run scripts/prepare_layoutlm_labels.py first.")
        sys.exit(1)
    with open(DATA_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_bbox(box, width, height):
    """LayoutLMv3 expects bboxes on a 0-1000 scale, normalized to image size."""
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]


def main() -> None:
    try:
        import torch
        from PIL import Image
        from transformers import (
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Processor,
            Trainer,
            TrainingArguments,
        )
        from datasets import Dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install torch transformers datasets "
              "accelerate seqeval  (see requirements-phase3.txt)")
        print("These are deliberately NOT in the main requirements.txt — "
              "they're training-time-only and GPU-oriented, not part of "
              "the served CPU inference image (see docs/01-ARCHITECTURE.md "
              "component boundaries).")
        sys.exit(1)

    records = load_records()
    print(f"Loaded {len(records)} labeled samples from {DATA_PATH}")
    if len(records) < 50:
        print(
            f"⚠️  n={len(records)} is far below the 100-300 labeled "
            f"documents docs/02-DATA-STRATEGY.md §3 calls a usable "
            f"minimum. A model trained on this many samples will not "
            f"produce a meaningful F1 — this run is a pipeline smoke "
            f"test (does the training loop run end-to-end without "
            f"crashing?), not the real Phase 3 result. Do not record "
            f"its F1 in experiments.csv as the Phase 3 exit-criteria "
            f"number; label more real data first."
        )

    processor = LayoutLMv3Processor.from_pretrained(
        BASE_MODEL, apply_ocr=False
    )
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    image_dir = REPO / "data" / "sample_receipts"

    def encode(example):
        image = Image.open(image_dir / example["image"]).convert("RGB")
        width, height = image.size
        boxes = [normalize_bbox(b, width, height) for b in example["bboxes"]]
        encoding = processor(
            image,
            example["tokens"],
            boxes=boxes,
            word_labels=[LABEL2ID[l] for l in example["labels"]],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}

    dataset = Dataset.from_list(records).map(encode, remove_columns=records[0].keys())
    dataset.set_format("torch")

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUT_DIR / "checkpoints"),
        num_train_epochs=20,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_OUT_DIR)
    processor.save_pretrained(MODEL_OUT_DIR)
    print(f"\nSaved fine-tuned model to {MODEL_OUT_DIR}")
    print("Next: python tests/test_phase3.py  (computes field-level F1 "
          "and compares against the Phase 2 baseline in experiments.csv)")


if __name__ == "__main__":
    main()
