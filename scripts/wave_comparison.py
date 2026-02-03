"""
Analizes and compares the wave of each part of the drumset
using the recorded samples at /data/raw/oneShot_drumset
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display

IN_DIR = "./data/raw/oneShot_drumset"
OUT_DIR = "./data/processed/wave_comparison"

os.makedirs(OUT_DIR, exist_ok=True)

N_FFT = 2048
HOP = 256

paths = sorted([os.path.join(IN_DIR, f) for f in os.listdir(IN_DIR) if f.lower().endswith(".wav")])
if not paths:
    raise RuntimeError(f"No audio found in {IN_DIR}")

names = []
signals = []
sr_used = None

for p in paths:
    name = os.path.splitext(os.path.basename(p))[0]
    y, sr = librosa.load(p, sr=None, mono=True)
    y = librosa.util.normalize(y)
    names.append(name)
    signals.append(y)
    sr_used = sr

min_len = min(len(y) for y in signals)
signals = [y[:min_len] for y in signals]

print(f"Loaded {len(signals)} WAVs | sr={sr_used} | dur={min_len/sr_used:.2f}s")

# time axis
t = np.arange(min_len) / sr_used

# plot waveforms
plt.figure(figsize=(12, 6))
offsets = np.arange(len(signals))[::-1] * 2.2  # vertical separation

for i, y in enumerate(signals):
    plt.plot(t, y + offsets[i], linewidth=1)

plt.yticks(offsets, names)
plt.xlabel("Time (s)")
plt.title("Aligned waveforms (stacked)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_waveforms_stacked.png"), dpi=200)
plt.close()

# individual features
rows = []

for name, y in zip(names, signals):
    # --- temporal features
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=HOP))**2
    freqs = librosa.fft_frequencies(sr=sr_used, n_fft=4096)

    def band_energy(fmin, fmax):
        fmax = min(fmax, freqs[-1])
        mask = (freqs >= fmin) & (freqs < fmax)
        return float(S[mask, :].sum()) if np.any(mask) else 0.0

    e_low = band_energy(20, 200)
    e_mid = band_energy(200, 2000)
    e_high = band_energy(2000, 12000)
    e_tot = e_low + e_mid + e_high + 1e-12

    # --- standard spectral features
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr_used)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr_used)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr_used, roll_percent=0.85)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    rows.append({
        "instrument": name,
        "band_low_ratio": e_low / e_tot,
        "band_mid_ratio": e_mid / e_tot,
        "band_high_ratio": e_high / e_tot,
        "spectral_centroid_hz": centroid, #low or high note
        "spectral_bandwidth_hz": bandwidth, #how wide is the spectrum
        "spectral_rolloff_hz": rolloff, #frequency below which 85% of energy is contained
        "spectral_flatness": flatness, #tonal vs noisy, 1 = noisy
        "zcr": zcr, # noisiness, how often it crosses zero
    })

df = pd.DataFrame(rows).sort_values("instrument").reset_index(drop=True)
df.to_csv(os.path.join(OUT_DIR, "features.csv"), index=False)
print("Saved:", os.path.join(OUT_DIR, "features.csv"))

# plot spectrograms
for name, y in zip(names, signals):
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr_used, hop_length=HOP, x_axis="time", y_axis="hz")
    plt.colorbar(label="dB")
    plt.title(f"Spectrogram: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"02_spectrogram_{name}.png"), dpi=200)
    plt.close()

feature_cols = [
    "band_low_ratio", "band_mid_ratio", "band_high_ratio",
    "spectral_centroid_hz", "spectral_rolloff_hz",
    "zcr", "spectral_flatness",
]

X = df[feature_cols].to_numpy(float)
Xz = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)

plt.figure(figsize=(14, max(4, 0.5 * len(df))))
im = plt.imshow(Xz, aspect="auto", interpolation="nearest", cmap="coolwarm")
cbar = plt.colorbar(im)
cbar.set_label("z-score (relative to feature mean)")
plt.yticks(np.arange(len(df)), df["instrument"].tolist())
plt.xticks(np.arange(len(feature_cols)), feature_cols, rotation=90)
plt.title("Feature heatmap (z-score) — relative comparison between instruments")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_feature_heatmap.png"), dpi=200)
plt.close()

print("Plots saved in:", OUT_DIR)
print(" - 01_waveforms_stacked.png")
print(" - 02_spectrogram_<instrument>.png")
print(" - 03_feature_heatmap.png")

# definition of grouped features

KICK_GROUP = ["36_kick"]
XSTICK_GROUP = ["37_xstick"]
SNARE_GROUP = ["38_snareHead", "40_snareRim"]
CLOSED_HIHAT_GROUP = ["42_hhClosed"]
PEDAL_HIHAT_GROUP = ["44_hhPedal"]
OPEN_HIHAT_GROUP = ["46_hhOpen"]
TOM_GROUP = ["45_tom2", "47_tom2Rim", "48_tom1", "50_tom1Rim"]
FLOOR_TOM_GROUP = ["43_tom3"]
CRASH_GROUP = ["49_crash1", "57_crash2"]
RIDE_BOW_GROUP = ["51_rideBow"]
RIDE_BELL_GROUP = ["53_rideBell"]
CHINESE_CYMBAL_GROUP = ["52_chinese"]
SPLASH_CYMBAL_GROUP = ["55_splash"]
VIBRASLAP_GROUP = ["58_vibraslap"]

GROUPS = {
    "Kick": KICK_GROUP,
    "XStick": XSTICK_GROUP,
    "Snare": SNARE_GROUP,
    "Closed_HiHat": CLOSED_HIHAT_GROUP,
    "Pedal_HiHat": PEDAL_HIHAT_GROUP,
    "Open_HiHat": OPEN_HIHAT_GROUP,
    "Tom": TOM_GROUP,
    "Floor_Tom": FLOOR_TOM_GROUP,
    "Crash": CRASH_GROUP,
    "Ride_Bow": RIDE_BOW_GROUP,
    "Ride_Bell": RIDE_BELL_GROUP,
    "Chinese_Cymbal": CHINESE_CYMBAL_GROUP,
    "Splash": SPLASH_CYMBAL_GROUP,
    "Vibraslap": VIBRASLAP_GROUP,
}

for group_name, instruments in GROUPS.items():

    group_data = [(n, y) for n, y in zip(names, signals) if n in instruments]

    # ignore groups with less than 2 instruments
    if len(group_data) < 2:
        continue

    plt.figure(figsize=(10, 4))

    for name, y in group_data:
        plt.plot(t, y, linewidth=1, label=name)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (normalized)")
    plt.title(f"Overlapped waveforms — {group_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"group_waveform_{group_name}.png"), dpi=200)
    plt.close()


for group_name, instruments in GROUPS.items():

    group_data = [(n, y) for n, y in zip(names, signals) if n in instruments]

    if len(group_data) < 2:
        continue

    plt.figure(figsize=(10, 3 * len(group_data)))

    for i, (name, y) in enumerate(group_data):
        D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        plt.subplot(len(group_data), 1, i + 1)
        librosa.display.specshow(
            S_db,
            sr=sr_used,
            hop_length=HOP,
            x_axis="time",
            y_axis="hz"
        )
        plt.title(name)
        if i < len(group_data) - 1:
            plt.xlabel("")
        plt.ylabel("Hz")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"group_spectrogram_{group_name}.png"), dpi=200)
    plt.close()
