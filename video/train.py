"""
Training script for visual speech recognition models.
Modified to work with:
- VisualDataset (returns dict)
- ResNet2DBiLSTM model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
import argparse
import time
import csv
from tqdm import tqdm

# Local imports
from data_utils.dataset_loader import get_data_loaders, MixupTransform
from models.resnet_lstm import ResNet2DBiLSTM
from config.config import load_config


# --------------------------------------------
# Training step
# --------------------------------------------
def train_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs, use_mixup=False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')

    for batch_idx, batch in enumerate(pbar):
        
        inputs = batch["lip_regions"].to(device)   # (B, C, T, H, W)
        targets = batch["label"].to(device)        # (B)

        # Standard mode
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        total_loss += loss.item()

        pbar.set_postfix({
            'loss': total_loss/(batch_idx+1),
            'acc': 100.*correct/total
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# --------------------------------------------
# Validation step
# --------------------------------------------
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating')

        for batch_idx, batch in enumerate(pbar):

            inputs = batch["lip_regions"].to(device)
            targets = batch["label"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_loss += loss.item()

            pbar.set_postfix({
                'loss': total_loss/(batch_idx+1),
                'acc': 100.*correct/total
            })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# --------------------------------------------
# Save checkpoint
# --------------------------------------------
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

# --------------------------------------------
# Log metrics to CSV
# --------------------------------------------
def log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, log_file):
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'timestamp'])
        writer.writerow([
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            time.strftime('%Y-%m-%d %H:%M:%S')
        ])

# --------------------------------------------
# Argument parsing
# --------------------------------------------
def get_args():
    args = {}
    args['config'] = "/home/aswath/Projects/capstone/multimodel_lipread/video/config/visual_config.yaml"
    args['resume'] = None
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args

# --------------------------------------------
# Main training loop
# --------------------------------------------
def main():
    args = get_args()

    # Load configuration
    config = load_config(args['config'])

    # Create output directory
    os.makedirs(config.get('training.save_dir'), exist_ok=True)

    # Load data
    train_loader, val_loader, test_loader = get_data_loaders(args['config'])
    num_classes = len(train_loader.dataset.classes)

    # Initialize model
    model = ResNet2DBiLSTM(
        num_classes=num_classes,
        config=config
    ).to(args['device'])

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('training.learning_rate'),
        weight_decay=config.get('training.weight_decay', 1e-5)
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Resume
    start_epoch = 0
    best_val_acc = 0.0

    if args['resume']:
        checkpoint = torch.load(args['resume'])
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        print(f"=> Resumed from checkpoint {args['resume']}")

    log_file = os.path.join(config.get('training.save_dir'), 'training_log.csv')

    # --------------------------------------------
    # EPOCH LOOP
    # --------------------------------------------
    for epoch in range(start_epoch, config.get('training.epochs')):

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            args['device'], epoch, config.get('training.epochs')
        )

        val_loss, val_acc = validate(model, val_loader, criterion, args['device'])

        scheduler.step(val_acc)

        log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, log_file)

        is_best = val_acc > best_val_acc
        best_val_acc = max(val_acc, best_val_acc)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_acc': best_val_acc,
            'optimizer': optimizer.state_dict(),
            'config': config
        }

        save_checkpoint(checkpoint, os.path.join(config.get('training.save_dir'), 'checkpoint.pth.tar'))

        if is_best:
            save_checkpoint(checkpoint, os.path.join(config.get('training.save_dir'), 'model_best.pth.tar'))

    # --------------------------------------------
    # Final test evaluation
    # --------------------------------------------
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, args['device'])
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    with open(os.path.join(config.get('training.save_dir'), 'test_results.txt'), 'w') as f:
        f.write(f'Test Loss: {test_loss:.4f}\n')
        f.write(f'Test Accuracy: {test_acc:.2f}%\n')
        f.write(f'Best Validation Accuracy: {best_val_acc:.2f}%\n')

if __name__ == '__main__':
    main()
