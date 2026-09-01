import torch
import librosa

#load
waveform, sample_rate = librosa.load('../data/raw/human/wavs/LJ001-0001.wav')
#transform
mel_spectro = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128)
#convert to tensor and add audio channel dimension
spectro = torch.from_numpy(mel_spectro).unsqueeze(0)

print(f'Mel Spectrogram shape: {spectro.shape}')