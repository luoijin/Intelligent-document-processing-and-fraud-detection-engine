import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

# 1. Setup Paths
REPO = Path(__file__).parent.parent
DATA_DIR = REPO / "data" / "layoutlm"
IMAGE_DIR = REPO / "data" / "sample_receipts"
OUTPUT_DIR = REPO / "models" / "layoutlmv3" / "v1"

# 2. Label Map
label_list = ["O", "B-MERCHANT", "I-MERCHANT", "B-DATE", "I-DATE", "B-TOTAL", "I-TOTAL"]
my_id2label = {i: l for i, l in enumerate(label_list)}
my_label2id = {l: i for i, l in enumerate(label_list)}

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

def normalize_bbox(bbox, width, height):
    return [
        max(0, min(1000, int(1000 * (bbox[0] / width)))),
        max(0, min(1000, int(1000 * (bbox[1] / height)))),
        max(0, min(1000, int(1000 * (bbox[2] / width)))),
        max(0, min(1000, int(1000 * (bbox[3] / height)))),
    ]

# 3. PyTorch Dataset (unchanged from your working version)
class SROIEPyTorchDataset(Dataset):
    def __init__(self, jsonl_path, image_dir, processor):
        self.processor = processor
        self.image_dir = Path(image_dir)
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = self.image_dir / item["image_file"]
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        words = item["tokens"]
        labels = item.get("ner_tags") or item.get("labels") or []
        bboxes = item["bboxes"]

        if len(labels) == 0:
            labels = [0] * len(words)

        min_len = min(len(words), len(labels), len(bboxes))
        words = words[:min_len]
        labels = labels[:min_len]
        boxes = [normalize_bbox(box, w, h) for box in bboxes[:min_len]]

        encoding = self.processor(
            images=image,
            text=words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx < len(labels):
                aligned_labels.append(labels[word_idx])
            else:
                aligned_labels.append(-100)

        item_dict = {k: v.squeeze(0) for k, v in encoding.items()}
        item_dict["labels"] = torch.tensor(aligned_labels, dtype=torch.long)
        return item_dict

train_dataset = SROIEPyTorchDataset(
    jsonl_path=DATA_DIR / "train.jsonl",
    image_dir=IMAGE_DIR,
    processor=processor
)

# NEW: validation set, so training isn't flying blind
eval_dataset = SROIEPyTorchDataset(
    jsonl_path=DATA_DIR / "val.jsonl",
    image_dir=IMAGE_DIR,
    processor=processor
)

# 4. Model Setup
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(label_list),
    id2label=my_id2label,
    label2id=my_label2id
)

# 5. Training Configuration
# CHANGED: max_steps=500 -> num_train_epochs=15 (real epoch count for 438 samples,
# instead of the leftover smoke-test step cap). Added eval every epoch, and
# keep the checkpoint with the lowest validation loss rather than just the last one.
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=15,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForTokenClassification(processor.tokenizer),
)

print("Starting LayoutLMv3 training...")
trainer.train()
trainer.save_model(str(OUTPUT_DIR))
processor.save_pretrained(str(OUTPUT_DIR))
print(f"Model saved successfully to {OUTPUT_DIR}")
