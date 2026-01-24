##############################################################################
# Build an index file (JSONL) from a dataset of wav + midi files.            #
# The index file contains, for each track:                                   #
#   - id                                                                     #
#   - path to wav file                                                       #
#   - path to midi file                                                      #
#   - duration in seconds                                                    #
#   - onsets per drum class (according to a drum map found in /experiments)  #
##############################################################################

import argparse
import json
from pathlib import Path

import pretty_midi
import soundfile as sf


def load_drum_map(path: Path) -> dict[str, set[int]]:
    drum_map = json.loads(path.read_text(encoding="utf-8"))
    return {k: set(v) for k, v in drum_map.items()}


def wav_duration_sec(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    return float(info.frames) / float(info.samplerate)


def midi_to_onsets_by_class(midi_path: Path, drum_map: dict[str, set[int]]) -> dict[str, list[float]]:
    """
    Returns {class_name: [onset_seconds, ...]} using label is_drum from pretty_midi
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    # output dictionary
    onsets = {cls: [] for cls in drum_map.keys()}

    # drum appearances
    drum_instruments = [inst for inst in pm.instruments if inst.is_drum]

    for inst in drum_instruments:
        for note in inst.notes:
            pitch = int(note.pitch)
            t = float(note.start)

            # assign classes according to drum_map (experiments/drum_map.json)
            for cls, pitches in drum_map.items():
                if pitch in pitches:
                    onsets[cls].append(t)
                    break

    # sort
    for cls in onsets:
        times = sorted(onsets[cls])
        list = []
        eps = 1e-4
        for x in times:
            list.append(x)
        onsets[cls] = list

    return onsets


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    """
    Matches .wav and .mid files under root by their relative paths (without extensions).
    Returns list of (wav_path, midi_path) tuples.
    """
    wavs = {p.with_suffix("").as_posix(): p for p in root.rglob("*.wav")}
    mids = {p.with_suffix("").as_posix(): p for p in root.rglob("*.mid")}

    common = sorted(set(wavs.keys()) & set(mids.keys()))
    pairs = [(wavs[k], mids[k]) for k in common]
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--drum_map", type=str, required=True)
    ap.add_argument("--out", type=str, default="data/processed/index.jsonl")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    drum_map_path = Path(args.drum_map).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    drum_map = load_drum_map(drum_map_path)

    pairs = find_pairs(data_root)

    # stats of total events per class
    total_events = {cls: 0 for cls in drum_map.keys()}

    with out_path.open("w", encoding="utf-8") as f:
        for wav_path, midi_path in pairs:
            dur = wav_duration_sec(wav_path)
            onsets = midi_to_onsets_by_class(midi_path, drum_map)

            for cls, times in onsets.items():
                total_events[cls] += len(times)

            track_id = wav_path.relative_to(data_root).with_suffix("").as_posix().replace("/", "__")

            row = {
                "id": track_id,
                "wav": str(wav_path),
                "midi": str(midi_path),
                "duration_sec": dur,
                "onsets_sec": onsets
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Result path: {out_path}")
    print(f"Pairs wav/mid found: {len(pairs)}")
    print("Events per class:")
    for cls, n in sorted(total_events.items(), key=lambda x: -x[1]):
        print(f"  {cls:12s} {n}")


if __name__ == "__main__":
    main()
