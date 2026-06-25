from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
import pretty_midi
import soundfile as sf
from tqdm import tqdm



# Same config as in cnn
@dataclass
class MFCCConfig:
    sr: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 256
    n_mfcc: int = 40
    win_seconds: float = 1.0



DRUM_CLASS_MAP: Dict[str, List[int]] = {
  "kick": [36],
  "xstick": [37],
  "snare": [38, 40], 
  "hihat_pedal": [44],
  "hihat_closed": [42, 22],
  "hihat_open": [46, 26],
  "tom": [45, 47, 48, 50],
  "floor_tom": [43],
  "vibraslap": [58],
  "crash": [49, 57],
  "ride_bow": [51, 59],
  "ride_bell": [53],
  "chinese_cymbal": [52],
  "splash_cymbal": [55]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Groove / E-GMD into MFCCs + frame labels.")
    parser.add_argument("--audio-root", type=str, required=True,
                        help="Root directory that contains the WAV files referenced in e-gmd-v1.0.0.csv")
    parser.add_argument("--midi-root", type=str, default=None,
                        help="Root directory that contains the MIDI files. Defaults to --audio-root")
    parser.add_argument("--csv-path", type=str, required=True,
                        help="Path to e-gmd-v1.0.0.csv")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory where processed data will be written")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable train-set z-normalization of MFCCs")
    parser.add_argument("--save-audio", action="store_true",
                        help="Also store the resampled waveform as .npy")
    return parser.parse_args()


def build_note_to_class(class_map: Dict[str, List[int]]) -> Tuple[Dict[int, int], List[str]]:
    class_names = list(class_map.keys())
    note_to_class: Dict[int, int] = {}
    for class_idx, (_, midi_notes) in enumerate(class_map.items()):
        for midi_note in midi_notes:
            if midi_note not in note_to_class:
                note_to_class[midi_note] = class_idx
    return note_to_class, class_names


def safe_stem_from_relative_path(rel_path: str) -> str:
    stem = Path(rel_path).with_suffix("")
    return str(stem).replace("/", "__")


def compute_mfcc(audio_path: Path, cfg: MFCCConfig) -> Tuple[np.ndarray, np.ndarray, float]:
    audio, original_sr = sf.read(audio_path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if original_sr != cfg.sr:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=cfg.sr)
    audio = audio.astype(np.float32)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=cfg.sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
    ).astype(np.float32)

    duration = len(audio) / cfg.sr
    return audio, mfcc, duration


def midi_to_frame_labels(
    midi_path: Path,
    num_frames: int,
    cfg: MFCCConfig,
    note_to_class: Dict[int, int],
    num_classes: int,
) -> Tuple[np.ndarray, List[dict]]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    labels = np.zeros((num_frames, num_classes), dtype=np.uint8)
    events: List[dict] = []

    for instrument in pm.instruments:
        for note in instrument.notes:
            if note.pitch not in note_to_class:
                continue
            cls = note_to_class[note.pitch]
            frame = int(round(note.start * cfg.sr / cfg.hop_length))
            frame = min(max(frame, 0), num_frames - 1)
            labels[frame, cls] = 1
            events.append({
                "t": float(note.start),
                "frame": int(frame),
                "pitch": int(note.pitch),
                "velocity": int(note.velocity),
                "class": int(cls),
            })

    events.sort(key=lambda x: x["t"])
    return labels, events

def accumulate_train_stats(sum_vec: np.ndarray | None, sumsq_vec: np.ndarray | None, feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    feat64 = feat.astype(np.float64)
    local_sum = feat64.sum(axis=1)
    local_sumsq = np.square(feat64).sum(axis=1)
    frame_count = feat.shape[1]

    if sum_vec is None:
        sum_vec = local_sum
        sumsq_vec = local_sumsq
    else:
        sum_vec += local_sum
        sumsq_vec += local_sumsq

    return sum_vec, sumsq_vec, frame_count


def z_normalize_features(feat: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((feat - mean[:, None]) / std[:, None]).astype(np.float32)


def ensure_dirs(root: Path) -> None:
    for split in ["train", "validation", "test"]:
        (root / split / "mfcc").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        (root / split / "meta").mkdir(parents=True, exist_ok=True)
        (root / split / "audio").mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    cfg = MFCCConfig()
    audio_root = Path(args.audio_root)
    midi_root = Path(args.midi_root) if args.midi_root else audio_root
    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    normalize = not args.no_normalize

    df = pd.read_csv(csv_path)
    required_cols = {"split", "audio_filename", "midi_filename", "duration", "drummer", "session", "id", "style", "bpm", "beat_type", "time_signature"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    ensure_dirs(output_dir)
    note_to_class, class_names = build_note_to_class(DRUM_CLASS_MAP)

    train_sum = None
    train_sumsq = None
    train_frames_total = 0
    manifest_by_split: Dict[str, List[dict]] = {"train": [], "validation": [], "test": []}
    cached_features: Dict[str, np.ndarray] = {}

    print("Pass 1/2: computing MFCCs, labels, and training statistics...")
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        split = str(row.split)
        if split not in manifest_by_split:
            continue

        audio_rel = str(row.audio_filename)
        midi_rel = str(row.midi_filename)
        audio_path = audio_root / audio_rel
        midi_path = midi_root / midi_rel

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")

        stem = safe_stem_from_relative_path(audio_rel)
        audio, mfcc, duration = compute_mfcc(audio_path, cfg)
        labels, events = midi_to_frame_labels(
            midi_path=midi_path,
            num_frames=mfcc.shape[1],
            cfg=cfg,
            note_to_class=note_to_class,
            num_classes=len(class_names),
        )

        if split == "train" and normalize:
            train_sum, train_sumsq, local_frames = accumulate_train_stats(train_sum, train_sumsq, mfcc)
            train_frames_total += local_frames

        cached_features[stem] = mfcc
        labels_path = output_dir / split / "labels" / f"{stem}.npy"
        np.save(labels_path, labels)

        if args.save_audio:
            audio_path_out = output_dir / split / "audio" / f"{stem}.npy"
            np.save(audio_path_out, audio)

        example_meta = {
            "stem": stem,
            "split": split,
            "audio_relpath": audio_rel,
            "midi_relpath": midi_rel,
            "mfcc_path": f"{split}/mfcc/{stem}.npy",
            "label_path": f"{split}/labels/{stem}.npy",
            "audio_path": f"{split}/audio/{stem}.npy" if args.save_audio else None,
            "drummer": str(row.drummer),
            "session": str(row.session),
            "id": str(row.id),
            "style": str(row.style),
            "bpm": int(row.bpm),
            "beat_type": str(row.beat_type),
            "time_signature": str(row.time_signature),
            "duration": float(duration),
            "csv_duration": float(row.duration),
            "num_frames": int(mfcc.shape[1]),
            "num_mfcc": int(mfcc.shape[0]),
            "onsets": events,
        }
        manifest_by_split[split].append(example_meta)

        with open(output_dir / split / "meta" / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(example_meta, f, indent=2)

    if normalize:
        if train_frames_total == 0 or train_sum is None or train_sumsq is None:
            raise RuntimeError("No training frames found; cannot compute normalization statistics.")
        mean = train_sum / train_frames_total
        var = (train_sumsq / train_frames_total) - np.square(mean)
        std = np.sqrt(np.maximum(var, 1e-8))
    else:
        mean = np.zeros(cfg.n_mfcc, dtype=np.float64)
        std = np.ones(cfg.n_mfcc, dtype=np.float64)

    print("Pass 2/2: writing normalized MFCCs and manifests...")
    for split, items in manifest_by_split.items():
        manifest_path = output_dir / split / "index.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            for item in tqdm(items, desc=f"writing {split}"):
                feat = cached_features[item["stem"]]
                feat = z_normalize_features(feat, mean, std) if normalize else feat.astype(np.float32)
                np.save(output_dir / item["mfcc_path"], feat)
                manifest_file.write(json.dumps(item) + "\n")

    summary = {
        "config": asdict(cfg),
        "normalize_over_training_set": normalize,
        "mfcc_mean": mean.tolist(),
        "mfcc_std": std.tolist(),
        "class_names": class_names,
        "class_map": DRUM_CLASS_MAP,
        "counts": {split: len(items) for split, items in manifest_by_split.items()},
    }
    with open(output_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump({split: [item["stem"] for item in items] for split, items in manifest_by_split.items()}, f, indent=2)

    print("Done.")
    print(f"Output written to: {output_dir}")


if __name__ == "__main__":
    main()