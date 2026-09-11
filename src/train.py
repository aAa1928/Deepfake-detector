"""Training entry point for deepfake-audio classifiers."""
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from init_resnet import init_resnet18, device, AI_path, human_path
from spectro_dataset import SpectrogramDataset
from asv_dataset import ASVDataset
from asv5_dataset import ASV5Dataset
from pathlib import Path
from collections import Counter
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report


#train_ratio = 0.85

#spectro_dataset = SpectrogramDataset(AI_path, human_path)
train_dataset = ASV5Dataset('../data/processed/flac_T', '../data/metadata/ASVSpoof5.train.tsv')
val_dataset = ASV5Dataset('../data/processed/flac_D', '../data/metadata/ASVSpoof5.dev.track_1.tsv')
#dataset = ConcatDataset([spectro_dataset, asvdataset])

train_counts = Counter(train_dataset.metadata.values())

ai_count = train_counts[0.0]
human_count = train_counts[1.0]

total = ai_count+human_count

ai_weight = total/(2*ai_count)
human_weight = total/(2*human_count)

print(f'Training class counts: {train_counts}')
print(f'AI weight: {ai_weight:.4f}')
print(f'Human weight: {human_weight:.4f}')

'''print(f'Total dataset size: {len(dataset)} samples')
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size'''

#train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = init_resnet18()

weights = torch.tensor([ai_weight, human_weight], device=device)

criterion = torch.nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#training loop
num_epochs = 20

best_balanced_accuracy = 0.0

for epoch in range(num_epochs):
    print(f'Starting epoch {epoch+1}/{num_epochs}...')

    model.train()

    running_loss = 0.0

    for spectrogram, label in train_loader:
        spectrogram = spectrogram.float()

        spectrogram = spectrogram.to(device)
        label = label.long().to(device)

        optimizer.zero_grad()

        outputs = model(spectrogram)

        loss = criterion(outputs, label)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss/len(train_loader)

    model.eval()

    correct = 0
    total = 0

    actual_counts = Counter()
    prediction_counts = Counter()

    ai_correct = 0
    ai_total = 0
    human_correct = 0
    human_total = 0

    '''with torch.no_grad():
        for spectrogram, label in val_loader:
            spectrogram = spectrogram.float()

            spectrogram = spectrogram.to(device)
            label = label.long().to(device)

            outputs = model(spectrogram)

            predictions = outputs.argmax(dim=1)

            total += label.size(0)
            correct += (predictions == label).sum().item()
'''
    with torch.no_grad():
        for spectrogram, label in val_loader:
            spectrogram = spectrogram.float().to(device)
            label = label.long().to(device)

            outputs = model(spectrogram)
            predictions = outputs.argmax(dim=1)

            total += label.size(0)
            correct += (predictions==label).sum().item()

            actual_counts.update(label.cpu().tolist())
            prediction_counts.update(predictions.cpu().tolist())

            ai_mask = (label==0)
            ai_total += ai_mask.sum().item()
            ai_correct += ((predictions==0) & ai_mask).sum().item()

            human_mask = (label==1)
            human_total += human_mask.sum().item()
            human_correct += ((predictions==1) & human_mask).sum().item()

    val_accuracy = correct/total
    ai_accuracy = ai_correct/ai_total if ai_total>0 else 0
    human_accuracy = human_correct/human_total if human_total>0 else 0

    balanced_accuracy = (ai_accuracy+human_accuracy)/2

    if balanced_accuracy > best_balanced_accuracy:
        best_balanced_accuracy = balanced_accuracy

        torch.save(model.state_dict(), 'models/resnet18_best.pth')
        print(f'New best model! Balanced Accuracy: {balanced_accuracy:.4f}')

    print(f'Epoch {epoch+1} done.')
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    print(f'Actual labels: {actual_counts}')
    print(f'Predictions: {prediction_counts}')
    print(f'AI Accuracy: {ai_accuracy:.4f}')
    print(f'Human Accuracy: {human_accuracy:.4f}')
    print(f'Training Loss: {epoch_loss:.4f}.')

torch.save(model.state_dict(), Path('models/resnet18_deepfake_audio_new.pth'))
print('Model saved to models/resnet18_best.pth')