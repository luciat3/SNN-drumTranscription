# SNN-drumTranscription

Automatic drum transcription system that converts drum audio into symbolic onset events and drum tablature.
The project compares two neural approaches:

* **SNN**: a Spiking Neural Network trained with an adaptation of the **e-prop** learning rule.
* **CNN**: a convolutional baseline trained with standard backpropagation.

This repository was developed as part of a Final Degree Project focused on automatic drum transcription, onset detection, few-shot adaptation, and comparison between spiking and conventional neural architectures.

---

## Table of Contents

* [Overview](#overview)
* [Repository Structure](#repository-structure)
* [Datasets](#datasets)
* [Drum Classes](#drum-classes)
* [Installation](#installation)
* [Training](#training)
* [Few-Shot Fine-Tuning](#few-shot-fine-tuning)
* [Inference and Onset Extraction](#inference-and-onset-extraction)
* [Acknowledgements](#acknowledgements)

---

## Overview

The objective of this project is to detect drum hits from audio and convert them into a structured representation that can later be used to generate drum notation or tablature.

The system follows a typical audio transcription pipeline:

1. **Audio preprocessing**

   * Load drum audio recordings.
   * Convert audio into time-frequency representations such as spectrograms, log-mel spectrograms, or MFCCs.

2. **Onset detection and classification**

   * Detect the time positions where drum hits occur.
   * Classify each onset into a drum instrument class.

3. **Model training**

   * Train either a CNN model using backpropagation or an SNN model using an e-prop-inspired learning approach.

4. **Post-processing**

   * Convert model probabilities into onset events.
   * Apply thresholds and cleaning steps.
   * Export detected events as JSON, MIDI-like data, LilyPond score files, or text-based tablature.


## Repository Structure

```text
SNN-drumTranscription/
├── data/
│   └── raw/
│       ├── oneShot_drumset/       # One-shot drum samples for each MIDI drum class
│       └── samples/               # Custom recorded audio examples and test grooves
│
├── experiments/
│   ├── cnn/                       # CNN inference and transcription outputs
│   ├── snn/                       # SNN inference and transcription outputs
│   └── drum_map.json              # Mapping between drum labels and MIDI/classes
│
├── models/
│   ├── cnn/
│   │   ├── dataset.py             # CNN dataset utilities
│   │   ├── fewshot.py             # Few-shot utilities for CNN experiments
│   │   ├── model.py               # CNN architecture
│   │   ├── train.py               # CNN training script
│   │   └── runs/                  # CNN checkpoints and metrics
│   │
│   └── snn/
│       ├── alif.py                # Adaptive LIF neuron implementation
│       ├── dataset_GROOVE.py      # Dataset loader for Groove/E-GMD data
│       ├── dataset_TIMIT.py       # Dataset loader adapted from phonetic experiments
│       ├── eprop.py               # e-prop learning components
│       ├── model.py               # SNN architecture
│       ├── train_GROOVE.py        # SNN training on Groove/E-GMD
│       ├── train_TEST.py          # Test training script
│       ├── train_TIMIT.py         # Phonetic example adaptation
│       └── runs/                  # SNN checkpoints and metrics
│
├── scripts/
│   ├── build_index.py             # Builds dataset metadata/index files
│   ├── check_spectrogram.py       # Visualizes or checks spectrogram data
│   ├── finetune_cnn_rwc.py        # CNN fine-tuning using RWC data
│   ├── finetune_snn_rwc.py        # SNN fine-tuning using RWC data
│   ├── groove_processing.py       # Groove/E-GMD preprocessing utilities
│   ├── make_splits.py             # Train/validation/test split generation
│   ├── predict_to_onsets.py       # Converts predictions into onset events
│   ├── train_cnn_fewshot.py       # Few-shot CNN training
│   ├── train_snn_fewshot.py       # Few-shot SNN training
│   └── wave_comparison.py         # Audio waveform comparison utilities
│
├── requirements.txt
└── README.md
```

---

## Datasets

This project uses external datasets for training and evaluation. Due to licensing and size constraints, full datasets are not included in this repository.

### Expanded Groove MIDI Dataset

The main training dataset is the **Expanded Groove MIDI Dataset**, used for drum performance transcription experiments.

L. Callender, C. Hawthorne, and J. Engel, “Improving Perceptual Quality of Drum Transcription with the Expanded Groove MIDI Dataset”, 2020, [Dataset]. arXiv:2004.00188v5

### RWC Dataset

The **RWC dataset** is used for polyphonic fine-tuning experiments.

M. Goto, S. Balke and M. Mueller, “RWC Music Database”. Zenodo, Feb. 16, 2026. doi: 10.5281/zenodo.18656623.


### Included Samples

The repository includes a small set of custom recorded samples under:

```text
data/raw/samples/
```

These files are useful for quick inference tests and qualitative evaluation.

The repository also includes one-shot drum samples under:

```text
data/raw/oneShot_drumset/
```

These are used as reference sounds for class analysis and experimentation.

---

## Drum Classes

The one-shot drum set contains the following MIDI drum classes:

| MIDI note | Instrument     |
| --------: | -------------- |
|        36 | Kick           |
|        37 | Cross stick    |
|        38 | Snare head     |
|        40 | Snare rim      |
|        42 | Closed hi-hat  |
|        43 | Low tom        |
|        44 | Pedal hi-hat   |
|        45 | Mid tom        |
|        46 | Open hi-hat    |
|        47 | Mid tom rim    |
|        48 | High tom       |
|        49 | Crash 1        |
|        50 | High tom rim   |
|        51 | Ride bow       |
|        52 | Chinese cymbal |
|        53 | Ride bell      |
|        55 | Splash         |
|        57 | Crash 2        |
|        58 | Vibraslap      |

The active set of classes used during a specific experiment may depend on the configuration, dataset preprocessing, and `experiments/drum_map.json`.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd SNN-drumTranscription
```

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If CUDA is available, install a PyTorch version compatible with your GPU and CUDA version.

---

## Training

### CNN Training

The CNN model is implemented in:

```text
models/cnn/model.py
```

The main CNN training script is:

```text
models/cnn/train.py
```

Training outputs are saved under:

```text
models/cnn/runs/
```

Each run may include:

* `best.pt`: best model checkpoint.
* `metrics.csv`: epoch-level metrics.
* `metrics.jsonl`: detailed training logs.
* `best_test_metrics.json`: best test evaluation metrics.

### SNN Training

The SNN model is implemented in:

```text
models/snn/model.py
```

The e-prop-related components are implemented in:

```text
models/snn/eprop.py
```

The adaptive LIF neuron implementation is located in:

```text
models/snn/alif.py
```

The main SNN training script for Groove is:

```text
models/snn/train_GROOVE.py
```

Training outputs are saved under:

```text
models/snn/runs/
```

Each run may include:

* `best.pt` or `best_groove_window.pt`: best model checkpoint.
* `metrics_*.jsonl`: detailed metric logs.
* `metrics_pytorch_groove_window_epoch.csv`: epoch-level results.
* `metrics_pytorch_groove_window_per_class.csv`: per-class metrics.
* `run_config.json`: configuration used for a specific run.

---

## Few-Shot Fine-Tuning

The project includes few-shot experiments for both CNN and SNN models.

CNN few-shot training:

```bash
python scripts/train_cnn_fewshot.py
```

SNN few-shot training:

```bash
python scripts/train_snn_fewshot.py
```

Existing runs include 1-shot, 5-shot, and 10-shot configurations. These experiments are useful for evaluating how well the models adapt to new drum sounds or reduced training data.

---

## RWC Fine-Tuning

Polyphonic fine-tuning with the RWC dataset is supported through:

```bash
python scripts/finetune_cnn_rwc.py
python scripts/finetune_snn_rwc.py
```

These scripts are intended to adapt the models to more complex polyphonic drum audio.

---

## Inference and Onset Extraction

After training, model outputs can be converted into onset events using:

```bash
python scripts/predict_to_onsets.py
```

The inference process produces:

* Raw probability arrays.
* Thresholded onset events.
* Cleaned onset events.
* JSON event files.
* Score or tablature output files.

Example output files:

```text
events_raw.json
events_clean.json
probs.npy
out_score.ly
out_score.txt
```


---

## Acknowledgements

This project is based on an adaptation of the original e-prop implementation.

```text
G. Bellec et al., “A solution to the learning dilemma for recurrent networks of spiking neurons”, Nat Commun, vol. 11, num. 1, des. 2020, doi: 10.1038/s41467-020-17236-y.
```

The repository also includes an adaptation of the phonetic example using PyTorch, which served as a starting point for the SNN implementation.

