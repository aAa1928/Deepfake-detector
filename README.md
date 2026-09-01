# Deepfake Audio Detector

A PyTorch audio-forensics project for classifying authentic human speech versus AI-generated or voice-cloned speech.

The initial dataset plan uses **LJSpeech** for authentic speech and **WaveFake** for synthetic speech. The first milestone is an end-to-end raw-waveform baseline; later iterations can add RawNet2/SincNet-style architectures and stronger forensic evaluation.

## Project flow

```text
LJSpeech + WaveFake
        |
        v
scripts/prepare_data.py
        |
        v
metadata CSV
        |
        v
scripts/create_splits.py
        |
        +--> train.csv
        +--> val.csv
        +--> test.csv
        |
        v
src/dataset.py + src/preprocessing.py
        |
        v
PyTorch DataLoader
        |
        v
src/models/
        |
        v
src/train.py
        |
        v
checkpoint
      /   \
     v     v
 evaluate  inference/demo
```

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install the matching `torch`/`torchaudio` wheels using the official PyTorch installation command before installing the remaining requirements.

## Data layout

Do **not** commit LJSpeech or WaveFake to GitHub. Put downloaded audio under `data/raw/` locally.

```text
data/
├── raw/                 # gitignored
│   ├── real/
│   │   └── ljspeech/
│   └── fake/
│       └── wavefake/
├── metadata/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── samples/
```

Target preprocessing for the first baseline:

- mono audio
- 16 kHz sample rate
- fixed 4-second windows
- 64,000 waveform samples per example
- normalized waveform amplitude
- label `0 = real`, `1 = fake`

## Repository structure

```text
.
├── configs/
│   └── baseline.yaml
├── data/
│   ├── metadata/
│   └── samples/
├── notebooks/
├── src/
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── models/
│   │   ├── baseline.py
│   │   └── rawnet.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
├── scripts/
│   ├── prepare_data.py
│   └── create_splits.py
├── checkpoints/
├── results/
├── demo/
│   └── quiz.py
├── requirements.txt
└── README.md
```

## Initial milestones

1. Build metadata for LJSpeech and WaveFake.
2. Create fixed train/validation/test splits.
3. Implement deterministic audio preprocessing and a PyTorch `Dataset`.
4. Train a small 1D-CNN baseline on raw waveforms.
5. Evaluate accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
6. Replace/compare the baseline with a RawNet2/SincNet-style model.
7. Build an interactive human-vs-model real/fake quiz.
