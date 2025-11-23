"""
Training script for visual speech recognition models.
Modified to match logging/saving style of audio training script.
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
from data_utils.dataset_loader import get_data_loaders
from models.resnet_lstm import ResNet2DBiLSTM
from config.config import load_config


# ============================================================
# Logging utilities (MATCHING AUDIO TRAINING SCRIPT)
# ============================================================
def init_log_files(model_name):
    os.makedirs("./metrics", exist_ok=True)

    csv_path = f"./metrics/{model_name}_training_log.csv"
    txt_path = f"./metrics/{model_name}_training_log.txt"

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss", "train_acc",
                "val_loss", "val_acc",
                "test_loss", "test_acc"
            ])

    if not os.path.exists(txt_path):
        with open(txt_path, "w") as f:
            f.write("Training Log\n\n")


def log_to_files(model_name, epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc):
    # CSV
    with open(f"./metrics/{model_name}_training_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            train_loss, train_acc,
            val_loss, val_acc,
            test_loss, test_acc
        ])

    # TXT log
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



# ============================================================
# Training step
# ============================================================
def train_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{num_epochs} [Train]')

    for batch_idx, batch in enumerate(pbar):
        inputs = batch["lip_regions"].to(device)
        targets = batch["label"].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        total_loss += loss.item()

        pbar.set_postfix({
            "loss": total_loss / (batch_idx + 1),
            "acc": 100. * correct / total
        })

    return total_loss / len(dataloader), 100. * correct / total


# ============================================================
# Validation step
# ============================================================
def validate(model, dataloader, criterion, device, title="Validating"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=title)

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
                "loss": total_loss / (batch_idx + 1),
                "acc": 100. * correct / total
            })

    return total_loss / len(dataloader), 100. * correct / total



# ============================================================
# Checkpoint Saving (Now .pth instead of .tar)
# ============================================================
def save_checkpoint(state, filename):
    torch.save(state, filename)



# ============================================================
# Args
# ============================================================
def get_args():
    return {
        "config": "/home/aswath/Projects/capstone/multimodel_lipread/video/config/visual_config.yaml",
        "resume": None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }



# ============================================================
# Main training loop (updated)
# ============================================================
def main():
    args = get_args()

    config = load_config(args["config"])
    save_dir = config.get('training.save_dir')
    os.makedirs(save_dir, exist_ok=True)

    model_name = config.get("model.name", "visual_model")
    init_log_files(model_name)

    train_loader, val_loader, test_loader = get_data_loaders(args["config"])
    num_classes = len(train_loader.dataset.classes)

    model = ResNet2DBiLSTM(num_classes=num_classes, config=config).to(args["device"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get("training.learning_rate"),
        weight_decay=config.get("training.weight_decay", 1e-5)
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    start_epoch = 1
    best_val_acc = 0.0

    # Resume
    if args["resume"]:
        ckpt = torch.load(args["resume"])
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        best_val_acc = ckpt["best_val_acc"]
        start_epoch = ckpt["epoch"]
        print(f"=> Resumed from checkpoint: {args['resume']}")

    # =============================================
    # Training loop
    # =============================================
    for epoch in range(start_epoch, config.get("training.epochs") + 1):

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            args["device"], epoch, config.get("training.epochs")
        )

        val_loss, val_acc = validate(model, val_loader, criterion, args["device"], "Validating")

        scheduler.step(val_acc)

        # Evaluate on test set **each epoch** (matching audio script)
        test_loss, test_acc = validate(model, test_loader, criterion, args["device"], "Testing")

        log_to_files(model_name, epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)

        # Save best checkpoint
        is_best = val_acc > best_val_acc
        best_val_acc = max(best_val_acc, val_acc)

        ckpt = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_acc": best_val_acc
        }

        save_checkpoint(ckpt, os.path.join(save_dir, "checkpoint.pth"))

        if is_best:
            save_checkpoint(ckpt, os.path.join(save_dir, "model_best.pth"))


    # =============================================
    # Final test after loading best model
    # =============================================
    best_ckpt = torch.load(os.path.join(save_dir, "model_best.pth"))
    model.load_state_dict(best_ckpt["state_dict"])

    test_loss, test_acc = validate(model, test_loader, criterion, args["device"], "Final Test")

    print(f"\nFinal Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    log_final_results(model_name, test_loss, test_acc)

    with open(os.path.join(save_dir, "test_results.txt"), "w") as f:
        f.write(f"Final Test Loss: {test_loss:.4f}\n")
        f.write(f"Final Test Acc: {test_acc:.2f}%\n")
        f.write(f"Best Val Acc: {best_val_acc:.2f}%\n")



if __name__ == "__main__":
    main()
