from pathlib import Path
import torch

class SpectrogramDataset:
    def __init__(self, AI_path, human_path):
        self.AI_path = Path(AI_path)
        self.human_path = Path(human_path)

        self.file_list = self.AI_path.glob('*.pt')+self.human_path.glob('*.pt')
        self.labels = []

        for file in self.AI_path.glob('*.pt'):
            self.labels.append(0.0)
        for file in self.human_path.glob('*.pt'):
            self.labels.append(1.0)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index, type=None):
        if type=='AI':
            file_path = self.AI_path.glob('*.pt')[index]
            label = 0.0
        elif type=='human':
            file_path = self.human_path.glob('*.pt')[index]
            label = 1.0
        elif type==None:
            file_path = self.file_list[index]
            label = self.labels[index]
        else:
            raise ValueError("Invalid type. Must be 'AI', 'human', or None.")

        spectrogram = torch.load(file_path)

        if len(spectrogram.shape) == 2:
            spectrogram = spectrogram.unsqueeze(0)

        return spectrogram, label