from pathlib import Path
import torch
import csv

class ASV5Dataset:
    def __init__(self, asv_path, label_path):
        self.asv_path = Path(asv_path)
        self.file_list = list(self.asv_path.glob('*.pt'))

        self.metadata={}

        with open(label_path, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for line in reader:
                type = line[8]
                filename = line[1]
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