"""
Analitza i compara la forma d'ona de cada part de la bateria
usant les mostres enregistrades a /data/raw/oneShot_drumset.

Serveix per classificar els instruments en grups amb característiques
espectrals similars, per ajudar el model a aprendre agrupant classes
properes.
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

# Style for single-column article figures. Instrument/class names stay in English.
FIG_W = 7.0
LATEX_COLUMN_WIDTH = 3.35
DPI = 300
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 17,
})

FEATURE_LABELS_CA = {
    "band_low_ratio": "energia baixa",
    "band_mid_ratio": "energia mitjana",
    "band_high_ratio": "energia alta",
    "spectral_centroid_hz": "centroide espectral",
    "spectral_rolloff_hz": "roll-off espectral",
    "zcr": "ZCR",
    "spectral_flatness": "planitud espectral",
}

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

print(f"Carregats {len(signals)} WAVs | sr={sr_used} | durada={min_len/sr_used:.2f}s")

# time axis
t = np.arange(min_len) / sr_used

# plot waveforms
plt.figure(figsize=(FIG_W, max(4.8, 0.35 * len(signals))))
offsets = np.arange(len(signals))[::-1] * 2.2  # vertical separation

for i, y in enumerate(signals):
    plt.plot(t, y + offsets[i], linewidth=1)

plt.yticks(offsets, names)
plt.xlabel("Temps (s)")
plt.title("Formes d'ona alineades")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_waveforms_stacked.png"), dpi=DPI)
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
print("Desat:", os.path.join(OUT_DIR, "features.csv"))

# plot spectrograms
for name, y in zip(names, signals):
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure(figsize=(FIG_W, 4.2))
    librosa.display.specshow(S_db, sr=sr_used, hop_length=HOP, x_axis="time", y_axis="hz")
    cbar = plt.colorbar()
    cbar.set_label("dB", fontsize=15)
    cbar.ax.tick_params(labelsize=13)
    plt.xlabel("Temps")
    plt.ylabel("Freqüència (Hz)")
    plt.title(f"Espectrograma: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"02_spectrogram_{name}.png"), dpi=DPI)
    plt.close()

feature_cols = [
    "band_low_ratio", "band_mid_ratio", "band_high_ratio",
    "spectral_centroid_hz", "spectral_rolloff_hz",
    "zcr", "spectral_flatness",
]

X = df[feature_cols].to_numpy(float)
Xz = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)

plt.figure(figsize=(FIG_W, max(4.8, 0.42 * len(df))))
im = plt.imshow(Xz, aspect="auto", interpolation="nearest", cmap="coolwarm")
cbar = plt.colorbar(im)
cbar.set_label("puntuació z", fontsize=15)
cbar.ax.tick_params(labelsize=13)
plt.yticks(np.arange(len(df)), df["instrument"].tolist())
plt.xticks(
    np.arange(len(feature_cols)),
    [FEATURE_LABELS_CA[c] for c in feature_cols],
    rotation=45,
    ha="right",
)
plt.title("Mapa de característiques")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_feature_heatmap.png"), dpi=DPI)
plt.close()

print("Gràfics desats a:", OUT_DIR)
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


def plot_hihat_spectrogram_comparison():
    hihat_specs = [
        ("42_hhClosed", "Hi-hat tancat"),
        ("44_hhPedal", "Hi-hat de pedal"),
        ("46_hhOpen", "Hi-hat obert"),
    ]
    signal_by_name = dict(zip(names, signals))
    missing = [name for name, _ in hihat_specs if name not in signal_by_name]
    if missing:
        print("No es genera la comparació de hi-hat; falten:", ", ".join(missing))
        return

    fig, axes = plt.subplots(
        len(hihat_specs),
        1,
        figsize=(LATEX_COLUMN_WIDTH, 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None

    for i, (name, display_name) in enumerate(hihat_specs):
        y = signal_by_name[name]
        D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        image = librosa.display.specshow(
            S_db,
            sr=sr_used,
            hop_length=HOP,
            x_axis="time",
            y_axis="hz",
            vmin=-80,
            vmax=0,
            ax=axes[i],
        )
        axes[i].set_title(display_name, fontsize=8.5, pad=3)
        axes[i].set_ylabel("Freq. (Hz)", fontsize=7)
        axes[i].tick_params(
            axis="both",
            labelsize=6.5,
            labelbottom=i == len(hihat_specs) - 1,
        )
        if i < len(hihat_specs) - 1:
            axes[i].set_xlabel("")

    axes[-1].set_xlabel("Temps (s)", fontsize=7)
    fig.suptitle("Espectrogrames de hi-hat", fontsize=9, y=1.02)
    cbar = fig.colorbar(image, ax=axes, pad=0.02, aspect=28)
    cbar.set_label("dB", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)

    output_path = os.path.join(OUT_DIR, "04_hihat_spectrogram_comparison.png")
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        os.path.splitext(output_path)[0] + ".pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)
    print("Desat:", output_path)


plot_hihat_spectrogram_comparison()

GROUPS = {
    "kick": KICK_GROUP,
    "xstick": XSTICK_GROUP,
    "snare": SNARE_GROUP,
    "hihat_closed": CLOSED_HIHAT_GROUP,
    "hihat_pedal": PEDAL_HIHAT_GROUP,
    "hihat_open": OPEN_HIHAT_GROUP,
    "tom": TOM_GROUP,
    "floor_tom": FLOOR_TOM_GROUP,
    "crash": CRASH_GROUP,
    "ride_bow": RIDE_BOW_GROUP,
    "ride_bell": RIDE_BELL_GROUP,
    "chinese_cymbal": CHINESE_CYMBAL_GROUP,
    "splash_cymbal": SPLASH_CYMBAL_GROUP,
    "vibraslap": VIBRASLAP_GROUP,
}

for group_name, instruments in GROUPS.items():

    group_data = [(n, y) for n, y in zip(names, signals) if n in instruments]

    # ignore groups with less than 2 instruments
    if len(group_data) < 2:
        continue

    plt.figure(figsize=(FIG_W, 4.2))

    for name, y in group_data:
        plt.plot(t, y, linewidth=1, label=name)

    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitud normalitzada")
    plt.title(f"Formes d'ona superposades: {group_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"group_waveform_{group_name}.png"), dpi=DPI)
    plt.close()


for group_name, instruments in GROUPS.items():

    group_data = [(n, y) for n, y in zip(names, signals) if n in instruments]

    if len(group_data) < 2:
        continue

    plt.figure(figsize=(FIG_W, 3.2 * len(group_data)))

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
        else:
            plt.xlabel("Temps")
        plt.ylabel("Freq. (Hz)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"group_spectrogram_{group_name}.png"), dpi=DPI)
    plt.close()
