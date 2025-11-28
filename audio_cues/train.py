# multimodal/train_mm.py
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.dataset import MultimodalDataset, collate_fn
from configs.config import load_config

from models.test_model import MultimodalNet
from models.early_fusion_mobile import EarlyFusionAttentionMobile
from models.middle_fusion_mobile import MiddleFusionAttentionMobile
from models.late_fusion_mobile import LateFusionAttentionMobile
from models.early_fusion_resnet import EarlyFusionAttentionResNet
from models.middle_fusion_resnet import MiddleFusionAttentionResNet
from models.late_fusion_resnet import LateFusionAttentionResNet


device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Logging Utilities
# -----------------------------
def init_log_files(model_name):
    os.makedirs("/home/aswath/Projects/capstone/multimodel_lipread/audio_cues/metrics", exist_ok=True)
    if not os.path.exists(f"/home/aswath/Projects/capstone/multimodel_lipread/audio_cues/metrics/{model_name}_training_log.csv"):
        with open(f"/home/aswath/Projects/capstone/multimodel_lipread/audio_cues/metrics/{model_name}_training_log.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss", "train_acc",
                "val_loss", "val_acc",
                "test_loss", "test_acc"
            ])


def log_to_files(model_name, epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc):
    # CSV log
    with open(f"/home/aswath/Projects/capstone/multimodel_lipread/audio_cues/metrics/{model_name}_training_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc])

    # Text log
    with open(f"/home/aswath/Projects/capstone/multimodel_lipread/audio_cues/metrics/{model_name}_training_log.txt", "a") as f:
        f.write(
            f"Epoch {epoch}\n"
            f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n"
            f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%\n"
            f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%\n\n"
        )


def log_final_results(model_name, test_loss, test_acc):
    with open(f"/home/aswath/Projects/capstone/multimodel_lipread/audio_cues/metrics/{model_name}_training_log.txt", "a") as f:
        f.write(f"Final Test Loss: {test_loss:.4f}, Final Test Acc: {test_acc:.2f}%\n")


# -----------------------------
# Epoch Functions
# -----------------------------
def run_epoch(model, loader, criterion, optimizer=None, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0

    desc = "Training" if train else "Validating"
    pbar = tqdm(loader, desc=desc)
    for mel, cue, label in pbar:
        mel, cue, label = mel.to(device), cue.to(device), label.to(device)

        if train:
            optimizer.zero_grad()

        out = model(mel, cue)
        loss = criterion(out, label)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * label.size(0)
        pred = out.argmax(1)
        total += label.size(0)
        correct += pred.eq(label).sum().item()

        pbar.set_postfix({
            "loss": total_loss/total,
            "acc": 100.*correct/total
        })

    return total_loss / total, 100. * correct / total


# -----------------------------
# Main Training Function
# -----------------------------
def main(cfg):
    model_name = "middle_fusion_resnet"
    os.makedirs("/home/aswath/Projects/capstone/multimodel_lipread/audio_cues/models_trained", exist_ok=True)
    init_log_files(model_name)

    # Load datasets
    train_ds = MultimodalDataset(cfg.get("dataset.root_dir"), cfg.get("dataset.cue_root"),
                                 cfg.get("dataset.input_size"), split="train")
    val_ds   = MultimodalDataset(cfg.get("dataset.root_dir"), cfg.get("dataset.cue_root"),
                                 cfg.get("dataset.input_size"), split="val")
    test_ds  = MultimodalDataset(cfg.get("dataset.root_dir"), cfg.get("dataset.cue_root"),
                                 cfg.get("dataset.input_size"), split="test")

    train_loader = DataLoader(train_ds, batch_size=cfg.get("train.batch"), shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=cfg.get("train.batch"), shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=cfg.get("train.batch"), shuffle=False, collate_fn=collate_fn)

    # Model
    cue_dim = next(iter(train_ds.desc2vec.values())).shape[0]  # dynamic cue dim

    if model_name == "early_fusion_mobile":
        model = EarlyFusionAttentionMobile(num_classes=cfg.get("dataset.num_classes"), cue_dim=cue_dim).to(device)
    elif model_name == "middle_fusion_mobile":
        model = MiddleFusionAttentionMobile(num_classes=cfg.get("dataset.num_classes"), cue_dim=cue_dim).to(device)
    elif model_name == "late_fusion_mobile":
        model = LateFusionAttentionMobile(num_classes=cfg.get("dataset.num_classes"), cue_dim=cue_dim).to(device)
    elif model_name == "early_fusion_resnet":
        model = EarlyFusionAttentionResNet(num_classes=cfg.get("dataset.num_classes"), cue_dim=cue_dim).to(device)
    elif model_name == "middle_fusion_resnet":
        model = MiddleFusionAttentionResNet(num_classes=cfg.get("dataset.num_classes"), cue_dim=cue_dim).to(device)
    elif model_name == "late_fusion_resnet":
        model = LateFusionAttentionResNet(num_classes=cfg.get("dataset.num_classes"), cue_dim=cue_dim).to(device)
    else:
        model = MultimodalNet(num_classes=cfg.get("dataset.num_classes"), cue_dim=cue_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("train.lr"))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_acc = 0
    epochs = cfg.get("train.epochs")

    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, train=True)
        val_loss, val_acc     = run_epoch(model, val_loader, criterion, train=False)

        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f'/home/aswath/Projects/capstone/multimodel_lipread/audio_cues/models_trained/{model_name}.pth')

        test_loss, test_acc = run_epoch(model, test_loader, criterion, train=False)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")

        log_to_files(model_name, epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)

    # Load best model and evaluate
    checkpoint = torch.load(f'/home/aswath/Projects/capstone/multimodel_lipread/audio_cues/models_trained/{model_name}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, train=False)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    log_final_results(model_name, test_loss, test_acc)


if __name__ == "__main__":
    cfg_path = "/home/aswath/Projects/capstone/multimodel_lipread/audio_cues/configs/ac_config.yaml"
    cfg = load_config(cfg_path)
    main(cfg)
