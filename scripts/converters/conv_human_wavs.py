from pathlib import Path
import numpy as np
import torch
import librosa
#establish paths
input = Path('../../data/raw/human/wavs')
output = Path('../../data/processed/human/wavs')
output.mkdir(exist_ok=True)

print('Starting .wav to spectrogram conversion...')

for index,file in enumerate(input.glob('*.wav')):
    #load
    waveform, sample_rate = librosa.load(file, sr=None)
    #transform
    mel_spectro = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128)
    mel_db=librosa.power_to_db(mel_spectro, ref=np.max)
    #convert to tensor and add audio channel dimension
    spectro = torch.from_numpy(mel_db).float().unsqueeze(0)
    #save
    name = f'spec_{file.stem}.pt'
    torch.save(spectro, output/name)

    print(f'Processed {file.name} to {name} with shape: {spectro.shape}')

print('Done!')
