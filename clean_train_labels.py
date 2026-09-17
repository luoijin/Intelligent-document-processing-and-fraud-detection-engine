"""
Removes training records where MERCHANT or TOTAL label-matching
failed entirely (silent negative-signal noise), and writes a
cleaned train.jsonl. Does NOT touch val.jsonl or test.jsonl --
those stay as-is for honest evaluation, missing labels and all.
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

path = Path("data/layoutlm/train.jsonl")
records = [json.loads(l) for l in open(path) if l.strip()]

kept, dropped = [], []
for r in records:
    labels = r.get("ner_tags") or r.get("labels")
    if has_entity(labels, "MERCHANT") and has_entity(labels, "TOTAL"):
        kept.append(r)
    else:
        dropped.append(r)

backup_path = path.with_suffix(".jsonl.bak")
path.rename(backup_path)
with open(path, "w") as f:
    for r in kept:
        f.write(json.dumps(r) + "\n")

print(f"Kept {len(kept)} records, dropped {len(dropped)}.")
print(f"Original file backed up to {backup_path}")
print(f"New train.jsonl written to {path}")