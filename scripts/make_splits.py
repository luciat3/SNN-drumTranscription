from pathlib import Path
import json
import random

index_path = Path("data/processed/RWC/index.jsonl")
out_path = Path("data/processed/RWC/splits.json")

items = []
with index_path.open("r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        items.append(item["id"])

random.seed(42)
random.shuffle(items)

n = len(items)
n_train = int(0.8 * n)
n_val = int(0.1 * n)

splits = {
    "train": items[:n_train],
    "val": items[n_train:n_train + n_val],
    "test": items[n_train + n_val:],
}

with out_path.open("w", encoding="utf-8") as f:
    json.dump(splits, f, indent=2)

print(f"Saved splits to {out_path}")
print({k: len(v) for k, v in splits.items()})