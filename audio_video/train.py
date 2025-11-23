import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils.dataset_av import GLipsMultimodalDataset
from config.config import load_config
from tqdm import tqdm
import csv
import os

from models.ef_cnn_lstm_resnet import create_early_fusion_resnet_model
from models.early_fusion import create_early_fusion_mobilenet_model
from models.late_fusion import create_late_fusion_mobilenet_model
from models.middle_fusion import create_mid_fusion_mobilenet_model
from models.early_fusion_fast import create_early_fusion_fast
from models.late_fusion_fast import create_late_fusion_fast
from models.middle_fusion_fast import create_mid_fusion_fast


# ----------------- Logging Functions -----------------
def init_log_files(model_name):
    if not os.path.exists(f"./metrics/{model_name}_training_log.csv"):
        with open(f"./metrics/{model_name}_training_log.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss", "train_acc",
                "val_loss", "val_acc",
                "test_loss", "test_acc"
            ])
    if not os.path.exists("./metrics/"):
        os.makedirs("./metrics/")


def log_to_files(model_name, epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc):
    # CSV log
    with open(f"./metrics/{model_name}_training_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc])

    # Text log
    with open(f"./metrics/{model_name}_training_log.txt", "a") as f:
        f.write(
            f"Epoch {epoch}\n"
            f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n"
            f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%\n"
            f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%\n\n"
        )


def log_final_results(model_name, test_loss, test_acc):
    with open(f"./metrics/{model_name}_training_log.txt", "a") as f:
        f.write(f"Final Test Loss: {test_loss:.4f}, Final Test Acc: {test_acc:.2f}%\n")


# ----------------- Training & Validation -----------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc="Training")
    for audio, video, labels in pbar:
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

        pbar.set_postfix({'loss': total_loss/len(loader), 'acc': 100.*correct/total})
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for audio, video, labels in tqdm(loader, desc="Validating"):
            audio, video, labels = audio.to(device), video.to(device), labels.to(device)
            outputs = model(audio, video)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return total_loss / len(loader), 100. * correct / total


# ----------------- Main Training Script -----------------
def main(config_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config(config_path)
    model_name = config.get('model.name')

    # Initialize logs
    init_log_files(model_name)

    # Load datasets
    train_dataset = GLipsMultimodalDataset(config.get('dataset.root_dir'), config.get('dataset.audio_input_size'))
    val_dataset = GLipsMultimodalDataset(config.get('dataset.root_dir'), config.get('dataset.audio_input_size'), split='val')
    test_dataset = GLipsMultimodalDataset(config.get('dataset.root_dir'), config.get('dataset.audio_input_size'), split='test')

    train_loader = DataLoader(train_dataset, batch_size=config.get('training.batch_size'), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('training.batch_size'), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.get('training.batch_size'), shuffle=False)

    # Model selection
    if model_name == 'early_fusion_resnet':
        model = create_early_fusion_resnet_model(config.get('dataset.num_classes'), config).to(device)
    elif model_name == 'early_fusion_mobilenet':
        model = create_early_fusion_mobilenet_model(config.get('dataset.num_classes'), config).to(device)
    elif model_name == 'late_fusion_mobilenet':
        model = create_late_fusion_mobilenet_model(config.get('dataset.num_classes'), config).to(device)
    elif model_name == 'middle_fusion_mobilenet':
        model = create_mid_fusion_mobilenet_model(config.get('dataset.num_classes'), config).to(device)
    elif model_name == 'early_fusion_fast':
        model = create_early_fusion_fast(config.get('dataset.num_classes'), config).to(device)
    elif model_name == 'late_fusion_fast':
        model = create_late_fusion_fast(config.get('dataset.num_classes'), config).to(device)
    elif model_name == 'middle_fusion_fast':
        model = create_mid_fusion_fast(config.get('dataset.num_classes'), config).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('training.learning_rate'))

    best_val_acc = 0.0

    # Training loop
    for epoch in range(1, config.get('training.epochs') + 1):
        print(f"\nEpoch {epoch}/{config.get('training.epochs')}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        test_loss, test_acc = validate(model, test_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")

        log_to_files(model_name, epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"./models_trained/{model_name}_best.pth")

    # Final test evaluation
    model.load_state_dict(torch.load(f"./models_trained/{model_name}_best.pth"))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Final Test Acc: {test_acc:.2f}%")
    log_final_results(model_name, test_loss, test_acc)


if __name__ == "__main__":
    main("/home/aswath/Projects/capstone/multimodel_lipread/audio_video/config/av_config.yaml")
