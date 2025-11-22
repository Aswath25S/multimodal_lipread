"""
Training script for visual speech recognition models.
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

# Import local modules
from data_utils.dataset_loader import get_data_loaders
from models.resnet_lstm import ResNet2DBiLSTM
from config.config import load_config

# Set up argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Train visual speech recognition model')
    parser.add_argument('--config', type=str, default='config/visual_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')
    return parser.parse_args()

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss/(batch_idx+1),
            'acc': 100.*correct/total
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
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

# Save checkpoint
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

# Log metrics to CSV
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

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    model_config = config['model']
    training_config = config['training']
    
    # Create output directory
    os.makedirs(training_config['save_dir'], exist_ok=True)
    
    # Set up data loaders
    train_loader, val_loader, test_loader = get_data_loaders(args.config)
    num_classes = len(train_loader.dataset.classes)
    
    # Initialize model
    model = VisualSpeechRecognitionModel(
        num_classes=num_classes,
        feature_dim=model_config.get('feature_dim', 512),
        hidden_size=model_config.get('hidden_size', 256),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.5)
    ).to(args.device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 1e-5)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> No checkpoint found at '{args.resume}'")
    
    # Training loop
    log_file = os.path.join(training_config['save_dir'], 'training_log.csv')
    
    for epoch in range(start_epoch, training_config['epochs']):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            args.device, epoch, training_config['epochs']
        )
        
        # Evaluate on validation set
        val_loss, val_acc = validate(model, val_loader, criterion, args.device)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Log metrics
        log_metrics(
            epoch, train_loss, train_acc, 
            val_loss, val_acc, log_file
        )
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        best_val_acc = max(val_acc, best_val_acc)
        
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_acc': best_val_acc,
            'optimizer': optimizer.state_dict(),
            'config': config
        }
        
        save_checkpoint(
            checkpoint, 
            os.path.join(training_config['save_dir'], 'checkpoint.pth.tar')
        )
        
        if is_best:
            save_checkpoint(
                checkpoint,
                os.path.join(training_config['save_dir'], 'model_best.pth.tar')
            )
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, args.device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Log final test results
    with open(os.path.join(training_config['save_dir'], 'test_results.txt'), 'w') as f:
        f.write(f'Test Loss: {test_loss:.4f}\n')
        f.write(f'Test Accuracy: {test_acc:.2f}%\n')
        f.write(f'Best Validation Accuracy: {best_val_acc:.2f}%\n')

if __name__ == '__main__':
    main()
