"""Final model evaluation on the held-out test split.

Planned metrics: accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
"""
import torch
from init_resnet import init_resnet18, device
from spectro_dataset import SpectrogramDataset
from asv_dataset import ASVDataset
from asv5_dataset import ASV5Dataset
from torch.utils.data import DataLoader, random_split
from collections import Counter
from sklearn.metrics import confusion_matrix

val_ratio = 0.15

#Split dataset
'''dataset = ASVDataset('../data/processed/asv', '../data/metadata/trial_metadata.txt')
train_size = int((1-val_ratio) * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)'''

val_dataset = ASVDataset('../data/processed/asv', '../data/metadata/trial_metadata.txt')

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load model
model = init_resnet18()
model.load_state_dict(torch.load('models/resnet18_best.pth'))
model.to(device)

model.eval()

ai_predictions = 0
human_predictions = 0

all_predictions = []
all_labels = []

correct = 0
total = 0

actual_counts = Counter()
prediction_counts = Counter()

with torch.no_grad():
    for spectrogram, label in val_loader:
        spectrogram = spectrogram.float().to(device)
        label = label.long().to(device)

        outputs = model(spectrogram)
        predictions = outputs.argmax(dim=1)

        total += label.size(0)
        correct += (predictions == label).sum().item()

        actual_counts.update(label.cpu().tolist())
        prediction_counts.update(predictions.cpu().tolist())

    val_accuracy = correct / total

    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print("Actual labels:", actual_counts)
    print("Predictions:", prediction_counts)
    val_accuracy = correct/total

'''with torch.no_grad():
    for spectrogram, label in val_loader:
        spectrogram = spectrogram.float()

        spectrogram = spectrogram.to(device)
        label = label.long().to(device)

        outputs = model(spectrogram)

        predictions = outputs.argmax(dim=1)

        ai_predictions += (predictions == 0).sum().item()
        human_predictions += (predictions == 1).sum().item()

        all_predictions.extend(predictions.tolist())
        all_labels.extend(label.tolist())

        total += label.size(0)
        correct += (predictions == label).sum().item()

    val_accuracy = correct/total

    print(f'Validation Accuracy: {val_accuracy:.4f}.')
    print("Predicted AI:", ai_predictions)
    print("Predicted Human:", human_predictions)'''

'''counts = Counter()

for _, label in val_dataset:
    counts[label] += 1

print(counts)'''