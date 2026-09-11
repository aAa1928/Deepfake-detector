from pathlib import Path
import torch

class SpectrogramDataset:
    def __init__(self, AI_path, human_path):
        self.AI_path = Path(AI_path)
        self.human_path = Path(human_path)

        self.file_list = list(self.AI_path.glob('*.pt'))+list(self.human_path.glob('*.pt'))
        self.labels = []

        for file in self.AI_path.glob('*.pt'):
            self.labels.append(0.0)
        for file in self.human_path.glob('*.pt'):
            self.labels.append(1.0)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index, type=None):
        if type=='AI':
            file_path = list(self.AI_path.glob('*.pt'))[index]
            label = 0.0
        elif type=='human':
            file_path = list(self.human_path.glob('*.pt'))[index]
            label = 1.0
        elif type==None:
            file_path = self.file_list[index]
            label = self.labels[index]
        else:
            raise ValueError("Invalid type. Must be 'AI', 'human', or None.")

        spectrogram = torch.load(file_path)

        target_width = 256

        if len(spectrogram.shape) == 2:
            spectrogram = spectrogram.unsqueeze(0)

        if spectrogram.shape[2] < target_width:
            padding = target_width - spectrogram.shape[2]
            spectrogram = torch.nn.functional.pad(spectrogram, (0, padding))
        elif spectrogram.shape[2] > target_width:
            spectrogram = spectrogram[:, :, :target_width]

        return spectrogram, label

    def get_human_list(self):
        return list(self.human_path.glob('*.pt'))
    def get_AI_list(self):
        return list(self.AI_path.glob('*.pt'))
    def get_file_list(self):
        return self.file_list