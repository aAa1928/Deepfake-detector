"""Shared audio preprocessing utilities.

Planned baseline pipeline:
1. decode audio
2. convert to mono
3. resample to 16 kHz
4. normalize waveform amplitude
5. crop or pad to a fixed 4-second window
"""

# TODO: implement reusable preprocessing functions used by training and inference.
