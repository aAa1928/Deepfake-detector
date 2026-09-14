"""Model architectures for deepfake audio detection."""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from spectro_dataset import SpectrogramDataset
from pathlib import Path
import random

AI_path = Path('../data/processed/AI/wavs')
human_path = Path('../data/processed/human/wavs')

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def init_resnet18():
    print(f'Using device: {device}')

    #initialize resnet18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    old_conv1 = model.conv1

    model.conv1 = nn.Conv2d(in_channels=1, out_channels=model.conv1.out_channels,
                            kernel_size=model.conv1.kernel_size, stride=model.conv1.stride,
                            padding=model.conv1.padding, bias=model.conv1.bias)

    with torch.no_grad():
        model.conv1.weight[:] = old_conv1.weight.mean(dim=1, keepdim=True)

    model.fc = nn.Linear(model.fc.in_features, 2) #gives 2 outputs(AI or human)

    '''add model to device(CUDA/MPS/CPU) AFTER ALL WEIGHTS AND CHANNEL 
    MODIFICATIONS ARE MADE, OTHERWISE IT WILL THROW AN ERROR'''
    model.to(device)

    return model

#test data compatibility with model
'''dataset = SpectrogramDataset(AI_path, human_path)
index = random.randint(0, dataset.__len__()-1)
test_item, test_label = dataset.__getitem__(index)
print(f'Data index: {index}')
#adding batch dimension and adding to CUDA/MPS/CPU
test_item = test_item.unsqueeze(0).to(device)
model = init_resnet18()
output = model(test_item)
print(f'Test output shape: {output.shape}, Actual label: {test_label}')
'''