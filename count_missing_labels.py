"""
Counts, across the full dataset, how many records are missing a
MERCHANT and/or TOTAL span entirely -- i.e. label-matching failed
silently for that record during the SROIE BIO-labeling step.
Run: python count_missing_labels.py
"""
import json
from pathlib import Path

id2label = {0: "O", 1: "B-MERCHANT", 2: "I-MERCHANT",
            3: "B-DATE", 4: "I-DATE", 5: "B-TOTAL", 6: "I-TOTAL"}

def has_entity(labels, entity):
    for lab in labels:
        lab_str = lab if isinstance(lab, str) else id2label.get(lab, "O")
        if lab_str in (f"B-{entity}", f"I-{entity}"):
            return True
    return False

for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
    path = Path("data/layoutlm") / split
    if not path.exists():
        continue
    records = [json.loads(l) for l in open(path) if l.strip()]
    n = len(records)
    missing_merchant = sum(
        1 for r in records
        if not has_entity(r.get("ner_tags") or r.get("labels"), "MERCHANT")
    )
    missing_date = sum(
        1 for r in records
        if not has_entity(r.get("ner_tags") or r.get("labels"), "DATE")
    )
    missing_total = sum(
        1 for r in records
        if not has_entity(r.get("ner_tags") or r.get("labels"), "TOTAL")
    )
    print(f"{split}: n={n}")
    print(f"  missing MERCHANT: {missing_merchant} ({100*missing_merchant/n:.1f}%)")
    print(f"  missing DATE:     {missing_date} ({100*missing_date/n:.1f}%)")
    print(f"  missing TOTAL:    {missing_total} ({100*missing_total/n:.1f}%)")
    print()