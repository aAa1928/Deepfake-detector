from pathlib import Path
import torch

class ASVDataset:
    def __init__(self, asv_path, label_path):
        self.asv_path = Path(asv_path)
        self.file_list = list(self.asv_path.glob('*.pt'))

        self.metadata={}

        with open(label_path, 'r') as f:
            for line in f:
                type = line.strip().split()[5]
                filename = line.strip().split()[1]
                label = 0.0 if type == 'spoof' else 1.0
                self.metadata[filename] = label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        label = self.metadata[file_path.stem]

        spectrogram = torch.load(file_path)

        target_width=256

        if len(spectrogram.shape) == 2:
            spectrogram = spectrogram.unsqueeze(0)

        if spectrogram.shape[2] < target_width:
            padding = target_width - spectrogram.shape[2]
            spectrogram = torch.nn.functional.pad(spectrogram, (0, padding))
        elif spectrogram.shape[2] > target_width:
            spectrogram = spectrogram[:, :, :target_width]

        return spectrogram, label