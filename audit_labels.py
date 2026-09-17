"""
Quick label-quality audit for data/layoutlm/train.jsonl.
Run locally: python audit_labels.py
Prints each sampled record's tokens with MERCHANT/TOTAL/DATE spans
highlighted, so you can eyeball whether the BIO tags actually land on
one contiguous, sensible span -- or are scattered across the receipt.
"""
import json
import random
from pathlib import Path

DATA_PATH = Path("data/layoutlm/train.jsonl")
N_SAMPLES = 15

id2label = {0: "O", 1: "B-MERCHANT", 2: "I-MERCHANT",
            3: "B-DATE", 4: "I-DATE", 5: "B-TOTAL", 6: "I-TOTAL"}

records = [json.loads(l) for l in open(DATA_PATH) if l.strip()]
random.seed(0)
sample = random.sample(records, min(N_SAMPLES, len(records)))

def spans_for(tokens, labels, entity):
    """Return list of (start_idx, end_idx, text) contiguous spans for entity."""
    spans = []
    cur = []
    cur_start = None
    for i, lab in enumerate(labels):
        lab_str = lab if isinstance(lab, str) else id2label.get(lab, "O")
        if lab_str == f"B-{entity}":
            if cur:
                spans.append((cur_start, i - 1, " ".join(cur)))
            cur = [tokens[i]]
            cur_start = i
        elif lab_str == f"I-{entity}" and cur:
            cur.append(tokens[i])
        else:
            if cur:
                spans.append((cur_start, i - 1, " ".join(cur)))
            cur = []
            cur_start = None
    if cur:
        spans.append((cur_start, len(tokens) - 1, " ".join(cur)))
    return spans

flagged = 0
for rec in sample:
    tokens = rec["tokens"]
    labels = rec.get("ner_tags") or rec.get("labels")
    rec_id = rec.get("id", rec.get("image_file", "?"))

    print(f"\n=== {rec_id} ===")
    for entity in ["MERCHANT", "DATE", "TOTAL"]:
        spans = spans_for(tokens, labels, entity)
        if not spans:
            print(f"  {entity}: (no span found)")
            continue
        if len(spans) > 1:
            flagged += 1
            print(f"  {entity}: *** {len(spans)} DISCONNECTED SPANS (likely bad) ***")
        for (s, e, text) in spans:
            gap_note = ""
            print(f"    span idx {s}-{e}: \"{text}\"")

print(f"\n\n{flagged} of {N_SAMPLES} sampled records have a multi-span "
      f"(disconnected) entity -- these are the ones most likely mislabeled "
      f"by the token-overlap fallback. Inspect them by hand.")