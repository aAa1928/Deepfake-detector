"""PyTorch Dataset for real and synthetic speech examples.

This module will read the train/validation/test metadata CSV files, load each
waveform, apply the shared preprocessing pipeline, and return `(waveform, label)`.
"""

# TODO: implement DeepfakeAudioDataset once dataset manifests are finalized.
