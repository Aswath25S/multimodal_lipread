import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils.dataset import GLipsDataset
from models.resnet_model import AudioResNet
from models.resnet_lstm_model import AudioResNetLSTM
from models.vgg_model import VGGAudioClassifier
from models.vgg_lstm_model import VGGWithLSTMClassifier
from models.lstm_resnet_model import LSTMResNet
from models.lstm_resnet_attn_model import DeepAudioNetWithAttention
from models.lstm_resnet_trans_model import LSTMResNetWithTransformer
from configs.config import load_config
from tqdm import tqdm
import argparse

import csv
import os

def init_log_files(model_name):
    # CSV log header
    if not os.path.exists(f"./metrics/{model_name}_training_log.csv"):
        with open(f"./metrics/{model_name}_training_log.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss", "train_acc",
                "val_loss", "val_acc",
                "test_loss", "test_acc"
            ])

def log_to_files(model_name, epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc):
    # CSV log
    with open(f"./metrics/{model_name}_training_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            train_loss, train_acc,
            val_loss, val_acc,
            test_loss, test_acc
        ])
    
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
        f.write(
            f"Final Test Loss: {test_loss:.4f}, Final Test Acc: {test_acc:.2f}%\n"
        )


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        specs = batch[0].to(device)
        labels = batch[1].to(device)
        
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': total_loss/len(dataloader), 'acc': 100.*correct/total})
    
    return total_loss/len(dataloader), 100.*correct/total

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            specs = batch[0].to(device)
            labels = batch[1].to(device)
            
            outputs = model(specs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss/len(dataloader), 100.*correct/total

def load_data(data_path, batch_size, input_size):
    train_dataset = GLipsDataset(data_path, input_size, split='train')
    val_dataset = GLipsDataset(data_path, input_size, split='val')
    test_dataset = GLipsDataset(data_path, input_size, split='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, val_loader, test_loader

def get_model(num_classes, input_size, model_name, version):
    if model_name == 'resnet':
        return AudioResNet(num_classes=num_classes)
    elif model_name == 'resnet_lstm':
        return AudioResNetLSTM(num_classes=num_classes)
    elif model_name == 'vgg':
        return VGGAudioClassifier(num_classes=num_classes, version=version)
    elif model_name == 'vgg_lstm':
        return VGGWithLSTMClassifier(num_classes=num_classes, version=version)
    elif model_name == 'lstm_resnet':
        return LSTMResNet(num_classes=num_classes, input_size=input_size)
    elif model_name == 'lstm_resnet_attn':
        return DeepAudioNetWithAttention(num_classes=num_classes, input_size=input_size)
    elif model_name == 'lstm_resnet_trans':
        return LSTMResNetWithTransformer(num_classes=num_classes, input_size=input_size)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

def main(config_path, device):
    config = load_config(config_path)
    data_path = config.get('dataset.root_dir')
    num_classes = config.get('dataset.num_classes')
    input_size = config.get('dataset.input_size')
    batch_size = config.get('training.batch_size')
    learning_rate = config.get('training.learning_rate')
    weight_decay = config.get('training.weight_decay')
    epochs = config.get('training.epochs')
    model_name = config.get('model.name')
    version = config.get('model.version')

    init_log_files(model_name)

    train_loader, val_loader, test_loader = load_data(data_path, batch_size, input_size)

    model = get_model(num_classes, input_size, model_name, version)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}/{epochs}')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f'./models_trained/{model_name}.pth')
        
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")

        log_to_files(model_name, epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)
    
    # Test
    model.load_state_dict(torch.load(f'./models_trained/{model_name}.pth')['model_state_dict'])
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f'\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')

    log_final_results(model_name, test_loss, test_acc)

if __name__ == '__main__':
    config_path = "/home/aswath/Projects/capstone/multimodel_lipread/audio/configs/audio_config.yaml"
    config = load_config(config_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(config_path, device)
    