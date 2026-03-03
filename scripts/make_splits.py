###############################################################################
# Create train/val/test splits from an index file using the official dataset  #
# splits defined in the e-gmd-v1.0.0.csv (split column: train/validation/test)#
###############################################################################

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# drummerX/<session>/<NUM>_something.wav  ->  drummerX/<session>/<NUM>
WAV_TO_CSV_ID_RE = re.compile(
    r"(?:^|/)(drummer[^/]+)/([^/]+)/(\d+)_", re.IGNORECASE
)


def read_index_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_csv_splits(csv_path: Path):
    """
    Reads e-gmd-v1.0.0.csv and returns:
      split_by_csv_id: dict[csv_id -> split]  where split in {"train","validation","test"}
    The CSV contains many rows per csv_id (different kits), but split is consistent per id.
    """
    import csv

    split_by_csv_id = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "id" not in reader.fieldnames or "split" not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain columns 'id' and 'split'. Found: {reader.fieldnames}"
            )

        for row in reader:
            csv_id = row["id"].strip()
            sp = row["split"].strip().lower()

            if sp not in {"train", "validation", "test"}:
                raise ValueError(f"Unexpected split value '{sp}' for id '{csv_id}'")

            prev = split_by_csv_id.get(csv_id)
            if prev is None:
                split_by_csv_id[csv_id] = sp
            elif prev != sp:
                raise ValueError(
                    f"Inconsistent split for id '{csv_id}': '{prev}' vs '{sp}'"
                )

    return split_by_csv_id


def wav_to_csv_id(wav_path: str) -> str | None:
    """
    Try to derive the CSV 'id' (drummer/session/NUM) from a wav path.
    Example:
      drummer1/eval_session/1_funk-groove1_138_beat_4-4_1.wav -> drummer1/eval_session/1
    """
    m = WAV_TO_CSV_ID_RE.search(wav_path or "")
    if not m:
        return None
    drummer, session, num = m.group(1), m.group(2), m.group(3)
    return f"{drummer}/{session}/{num}"


def extract_drummer_from_id_or_wav(item_id: str, wav_path: str) -> str:
    # Prefer csv-like id (starts with drummerX/...)
    if item_id:
        parts = item_id.split("/")
        if parts and parts[0].lower().startswith("drummer"):
            return parts[0]
    # Fallback to wav parsing
    csv_id = wav_to_csv_id(wav_path)
    if csv_id:
        return csv_id.split("/")[0]
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, default="data/processed/index.jsonl")
    ap.add_argument("--csv", type=str, default="data/raw/groove/e-gmd-v1.0.0.csv")
    ap.add_argument("--out", type=str, default="data/processed/splits.json")
    args = ap.parse_args()

    index_path = Path(args.index).resolve()
    csv_path = Path(args.csv).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = read_index_jsonl(index_path)
    split_by_csv_id = load_csv_splits(csv_path)

    train_ids, val_ids, test_ids = [], [], []
    per_drummer_stats = defaultdict(lambda: {"total": 0, "train": 0, "val": 0, "test": 0})

    missing = 0
    unresolved_examples = []

    for it in items:
        item_id = it.get("id", "")
        wav = it.get("wav", "")

        csv_id = item_id if item_id in split_by_csv_id else None
        if csv_id is None:
            csv_id = wav_to_csv_id(wav)

        if not csv_id or csv_id not in split_by_csv_id:
            missing += 1
            if len(unresolved_examples) < 10:
                unresolved_examples.append({"id": item_id, "wav": wav, "derived": csv_id})
            continue

        sp = split_by_csv_id[csv_id]  # train / validation / test
        drummer = extract_drummer_from_id_or_wav(item_id, wav)

        per_drummer_stats[drummer]["total"] += 1

        if sp == "train":
            train_ids.append(item_id)
            per_drummer_stats[drummer]["train"] += 1
        elif sp == "validation":
            val_ids.append(item_id)
            per_drummer_stats[drummer]["val"] += 1
        elif sp == "test":
            test_ids.append(item_id)
            per_drummer_stats[drummer]["test"] += 1

    splits = {
        "strategy": "dataset_defined_csv_splits",
        "index": str(index_path),
        "csv": str(csv_path),
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "total": len(train_ids) + len(val_ids) + len(test_ids),
            "missing_unmatched": missing,
        },
        "per_drummer": dict(sorted(per_drummer_stats.items())),
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    if missing > 0:
        splits["unmatched_examples"] = unresolved_examples

    out_path.write_text(json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nResult path {out_path}")
    print("\nSplit count:", splits["counts"])
    if missing:
        print(f"\nWARNING: {missing} items in index.jsonl could not be matched to CSV splits.")
        print("Examples (up to 10):")
        for ex in unresolved_examples:
            print(" ", ex)


if __name__ == "__main__":
    main()