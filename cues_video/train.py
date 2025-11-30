import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from data_utils.dataset import MultimodalCueVideoDataset
from models.test_model import MultimodalCueVideoNet

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Build label encoder
# -----------------------
def build_label_encoder(ds):
    words = sorted(list(set(s["word"] for s in ds.samples)))
    le = LabelEncoder()
    le.fit(words)
    return le


# -----------------------
# Collate function
# -----------------------
def collate_fn(batch):
    cues, videos, labels, sids = zip(*batch)
    return (
        torch.stack(cues),
        torch.stack(videos),
        list(labels),
        list(sids)
    )


# -----------------------
# One Epoch
# -----------------------
def run_epoch(model, loader, optimizer, criterion, le, train=True):
    total, correct, loss_sum = 0, 0, 0
    model.train() if train else model.eval()

    pbar = tqdm(loader, desc="Train" if train else "Eval")

    for cues, videos, labels, _ in pbar:
        cues = cues.to(device)
        videos = videos.to(device)
        y = torch.tensor(le.transform(labels), device=device)

        if train:
            optimizer.zero_grad()

        out = model(cues, videos)
        loss = criterion(out, y)

        if train:
            loss.backward()
            optimizer.step()

        loss_sum += loss.item() * y.size(0)
        pred = out.argmax(1)
        total += y.size(0)
        correct += (pred == y).sum().item()

        pbar.set_postfix({
            "loss": loss_sum / total,
            "acc": 100 * correct / total
        })

    return loss_sum / total, 100 * correct / total


# -----------------------
# MAIN TRAIN LOOP
# -----------------------
def main():
    cue_root = "/home/aswath/Projects/capstone/multimodel_lipread/cues"
    lip_root = "/home/aswath/Projects/capstone/GLips_4_lip_regions/lipread_files"
    batch = 4
    epochs = 30

    train_ds = MultimodalCueVideoDataset(cue_root, lip_root, split="train")
    val_ds = MultimodalCueVideoDataset(cue_root, lip_root, split="val")
    test_ds = MultimodalCueVideoDataset(cue_root, lip_root, split="test")

    le = build_label_encoder(train_ds)
    num_classes = len(le.classes_)
    cue_dim = next(iter(train_ds.desc2vec.values())).shape[0]

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False,
                             collate_fn=collate_fn, num_workers=4)

    model = MultimodalCueVideoNet(num_classes, cue_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, le, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, criterion, le, train=False)

        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")
        print(f"Val  : loss={val_loss:.4f}, acc={val_acc:.2f}%")

        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), "model_best.pth")
            print("✅ Model saved")

    print("\n🔍 Final evaluation on test set")
    model.load_state_dict(torch.load("model_best.pth"))
    test_loss, test_acc = run_epoch(model, test_loader, optimizer, criterion, le, train=False)
    print(f"Test Acc = {test_acc:.2f}%")


if __name__ == "__main__":
    main()
