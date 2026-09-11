from pathlib import Path
import numpy as np
import torch
import librosa
import subprocess
#establish paths
input = Path('../../data/raw/flac_T')
output = Path('../../data/processed/flac_T')
output.mkdir(exist_ok=True)

print('Starting .flac to spectrogram conversion...')

for index,file in enumerate(input.glob('*.flac')):
    result = subprocess.run(['ffmpeg', '-i', file, '-f', 'f32le', '-ac', '1', '-ar', '16000', 'pipe:1'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)

    audio = np.frombuffer(result.stdout, dtype=np.float32)

    mel_spectro = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128)
    mel_db = librosa.power_to_db(mel_spectro, ref=np.max)

    spectro = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
    #save
    name = f'{file.stem}.pt'
    torch.save(spectro, output/name)

    print(f'Processed {file.name} to {name} with shape: {spectro.shape}')

print('Done!')
