###############################################################################
# Create train/val/test splits from an index file, ensuring an 80-10-10       #
# strategy from every drummer so that the model is trained in every style     #
###############################################################################

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


DRUMMER_RE = re.compile(r"(?:^|/)(drummer[^/]+)(?:/|$)", re.IGNORECASE)


def read_index_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def extract_drummer(wav_path: str) -> str:
    m = DRUMMER_RE.search(wav_path)
    return m.group(1)


def split_counts(n: int):
    """
    Returns (n_train, n_val, n_test) following these rules:
      - n >= 10: 80/10/10
      - 3 <= n < 10: train=n-2, val=1, test=1
      - n == 2: train=1, val=0, test=1
      - n == 1: train=1, val=0, test=0
    """
    if n >= 10:
        n_train = int(round(n * 0.8))
        n_val = int(round(n * 0.1))
        n_test = n - n_train - n_val
        if n_val == 0:
            n_val = 1
            n_train -= 1
        if n_test == 0:
            n_test = 1
            n_train -= 1
        if n_train < 1:
            n_train = max(1, n - (n_val + n_test))
        return n_train, n_val, n_test

    if n >= 3:
        return n - 2, 1, 1
    if n == 2:
        return 1, 0, 1
    return 1, 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, default="data/processed/index.jsonl")
    ap.add_argument("--out", type=str, default="data/processed/splits.json")
    # seed allowing reproducibility of the splits
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    index_path = Path(args.index).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = read_index_jsonl(index_path)

    # group by drummer
    by_drummer = defaultdict(list)
    for it in items:
        wav = it.get("wav", "")
        drummer = extract_drummer(wav)
        by_drummer[drummer].append(it["id"])

    rng = random.Random(args.seed)

    train_ids, val_ids, test_ids = [], [], []
    per_drummer_stats = {}

    for drummer, ids in sorted(by_drummer.items()):
        ids = list(ids)
        rng.shuffle(ids)

        n = len(ids)
        n_train, n_val, n_test = split_counts(n)

        train_part = ids[:n_train]
        val_part = ids[n_train:n_train + n_val]
        test_part = ids[n_train + n_val:n_train + n_val + n_test]

        train_ids.extend(train_part)
        val_ids.extend(val_part)
        test_ids.extend(test_part)

        per_drummer_stats[drummer] = {
            "total": n,
            "train": len(train_part),
            "val": len(val_part),
            "test": len(test_part),
        }

    splits = {
        "seed": args.seed,
        "strategy": "drummer_stratified_80_10_10",
        "index": str(index_path),
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "total": len(train_ids) + len(val_ids) + len(test_ids),
        },
        "per_drummer": per_drummer_stats,
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    out_path.write_text(json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8")

    # imprimir resumen
    print(f"\nResult path {out_path}")
    print("\nSplit count:", splits["counts"])
    print("\nDrummer counts:")
    for drummer, st in per_drummer_stats.items():
        print(f"  {drummer:10s} total={st['total']:4d}  train={st['train']:4d}  val={st['val']:4d}  test={st['test']:4d}")


if __name__ == "__main__":
    main()
