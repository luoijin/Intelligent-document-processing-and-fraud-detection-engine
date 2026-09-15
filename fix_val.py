import json
from pathlib import Path

id2label = {0:"O",1:"B-MERCHANT",2:"I-MERCHANT",3:"B-DATE",4:"I-DATE",5:"B-TOTAL",6:"I-TOTAL"}

path = Path("data/layoutlm/val.jsonl")
fixed = []
for line in open(path):
    rec = json.loads(line)
    fixed.append({
        "image": rec["image_file"],
        "tokens": rec["tokens"],
        "bboxes": rec["bboxes"],
        "labels": [id2label[t] for t in rec["ner_tags"]],
    })
with open(path, "w") as f:
    for r in fixed:
        f.write(json.dumps(r) + "\n")
print(f"Fixed {len(fixed)} records")