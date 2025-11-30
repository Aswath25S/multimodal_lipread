import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from configs.config import load_config

from data_utils.dataset import MultimodalCueVideoDataset
from models.test_model import MultimodalCueVideoNet


device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Logging
# -----------------------------
def init_log_files(model_name):
    os.makedirs("./metrics", exist_ok=True)
    os.makedirs("./models_trained", exist_ok=True)

    csv_path = f"./metrics/{model_name}_training_log.csv"
    txt_path = f"./metrics/{model_name}_training_log.txt"

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"])

    if not os.path.exists(txt_path):
        with open(txt_path, "w") as f:
            f.write("Training Log\n\n")


def log_to_files(model_name, epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc):
    with open(f"./metrics/{model_name}_training_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc])

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


# -----------------------------
# Build label encoder
# -----------------------------
def build_label_encoder(ds):
    words = sorted(set(s["word"] for s in ds.samples))
    le = LabelEncoder()
    le.fit(words)
    return le


# -----------------------------
# Collate function
# -----------------------------
def collate_fn(batch):
    cues, videos, labels, sids = zip(*batch)
    return torch.stack(cues), torch.stack(videos), list(labels), list(sids)


# -----------------------------
# Train / Eval loops
# -----------------------------
def train_epoch(model, loader, criterion, optimizer, le):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc="Training")

    for cues, videos, labels, _ in pbar:
        cues = cues.to(device)
        videos = videos.to(device)
        targets = torch.tensor(le.transform(labels), device=device)

        optimizer.zero_grad()
        outputs = model(cues, videos)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        pred = outputs.argmax(1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix({
            "loss": total_loss / total,
            "acc": 100 * correct / total
        })

    return total_loss / total, 100 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, le):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc="Validating")

    for cues, videos, labels, _ in pbar:
        cues = cues.to(device)
        videos = videos.to(device)
        targets = torch.tensor(le.transform(labels), device=device)

        outputs = model(cues, videos)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * targets.size(0)
        pred = outputs.argmax(1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix({
            "loss": total_loss / total,
            "acc": 100 * correct / total
        })

    return total_loss / total, 100 * correct / total


# -----------------------------
# MAIN
# -----------------------------
def main(config):
    model_name = "cue_video_model"

    cue_root = config.get("dataset.cue_root")
    lip_root = config.get("dataset.lip_regions_root")

    batch_size = config.get("train.batch", 4)
    learning_rate = config.get("train.lr", 1e-4)
    epochs = config.get("train.epochs", 30)
    weight_decay = config.get("train.weight_decay", 1e-4)

    init_log_files(model_name)

    # Load datasets
    train_ds = MultimodalCueVideoDataset(cue_root, lip_root, split="train")
    val_ds = MultimodalCueVideoDataset(cue_root, lip_root, split="val")
    test_ds = MultimodalCueVideoDataset(cue_root, lip_root, split="test")

    # Label encoder
    le = build_label_encoder(train_ds)
    num_classes = len(le.classes_)
    cue_dim = next(iter(train_ds.desc2vec.values())).shape[0]

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Model
    model = MultimodalCueVideoNet(num_classes, cue_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val = 0

    # Training Loop
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, le)
        val_loss, val_acc = validate(model, val_loader, criterion, le)
        test_loss, test_acc = validate(model, test_loader, criterion, le)

        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")
        print(f"Val  : loss={val_loss:.4f}, acc={val_acc:.2f}%")
        print(f"Test : loss={test_loss:.4f}, acc={test_acc:.2f}%")

        scheduler.step(val_loss)

        log_to_files(model_name, epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)

        # Save best model
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc
            }, f"./models_trained/{model_name}.pth")
            print("✅ Best model saved")

    # Final Test
    print("\n🔍 Loading best model for final evaluation...")
    ckpt = torch.load(f"./models_trained/{model_name}.pth")
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc = validate(model, test_loader, criterion, le)
    print(f"Final Test Results: loss={test_loss:.4f}, acc={test_acc:.2f}%")

    log_final_results(model_name, test_loss, test_acc)


if __name__ == "__main__":
    config_path = "/home/aswath/Projects/capstone/multimodel_lipread/cues_video/configs/cv_config.yaml"
    config = load_config(config_path)
    main(config)
