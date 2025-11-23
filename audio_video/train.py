import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils.dataset_av import GLipsMultimodalDataset
from models.ef_cnn_lstm_resnet import create_early_fusion_model
from config.config import load_config
import os
from tqdm import tqdm

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for audio, video, labels in tqdm(loader):
        audio, video, labels = audio.to(device), video.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(audio, video)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return total_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for audio, video, labels in loader:
            audio, video, labels = audio.to(device), video.to(device), labels.to(device)
            outputs = model(audio, video)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return total_loss / len(loader), 100 * correct / total

def main(config_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config(config_path)

    train_dataset = GLipsMultimodalDataset(config.get('dataset.root_dir'), config.get('dataset.audio_input_size'))
    val_dataset = GLipsMultimodalDataset(config.get('dataset.root_dir'), config.get('dataset.audio_input_size'), split='val')
    test_dataset = GLipsMultimodalDataset(config.get('dataset.root_dir'), config.get('dataset.audio_input_size'), split='test')

    train_loader = DataLoader(train_dataset, batch_size=config.get('training.batch_size'), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('training.batch_size'), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.get('training.batch_size'), shuffle=False)

    model = create_early_fusion_model(config.get('dataset.num_classes'), config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('training.learning_rate'))

    best_val_acc = 0.0
    for epoch in range(1, config.get('training.epochs') + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        test_loss, test_acc = validate(model, test_loader, criterion, device)

        print(f"Epoch {epoch}: Train {train_acc:.2f}% | Val {val_acc:.2f}% | Test {test_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"./models_trained/audio_video_best.pth")

if __name__ == "__main__":
    main("/home/aswath/Projects/capstone/multimodel_lipread/audio_video/config/av_config.yaml")
