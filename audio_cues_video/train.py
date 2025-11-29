# multimodal/train_triple.py
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import load_config
from data_utils.dataset import MultimodalTripleDataset, collate_fn_triple

from models.test_model import MultimodalThreeNet
from models.early_fusion_mobile import MultimodalAttentionEarly
from models.middle_fusion_mobile import MultimodalAttentionMiddle
from models.late_fusion_mobile import MultimodalAttentionLate
from models.early_fusion_resnet import MultimodalAttentionEarlyResNet
from models.middle_fusion_resnet import MultimodalAttentionMiddleResNet
from models.late_fusion_resnet import MultimodalAttentionLateResNet

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
def init_log_files(model_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{model_name}_training_log.csv")
    txt_path = os.path.join(out_dir, f"{model_name}_training_log.txt")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"])
    if not os.path.exists(txt_path):
        with open(txt_path, "w") as f:
            f.write("Training Log\n\n")


def log_to_files(model_name, out_dir, epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc):
    with open(os.path.join(out_dir, f"{model_name}_training_log.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc])
    with open(os.path.join(out_dir, f"{model_name}_training_log.txt"), "a") as f:
        f.write(
            f"Epoch {epoch}\n"
            f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n"
            f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%\n"
            f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%\n\n"
        )


# -----------------------------
def run_epoch(model, loader, criterion, optimizer=None, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    desc = "Training" if train else "Validating"
    pbar = tqdm(loader, desc=desc)
    for batch in pbar:
        mel, cue, lips, label = batch
        mel = mel.to(device)
        cue = cue.to(device)
        lips = lips.to(device)
        label = label.to(device)

        if train:
            optimizer.zero_grad()

        out = model(mel, cue, lips)
        loss = criterion(out, label)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * label.size(0)
        pred = out.argmax(1)
        total += label.size(0)
        correct += pred.eq(label).sum().item()

        pbar.set_postfix({"loss": total_loss / total, "acc": 100. * correct / total})

    return total_loss / total, 100. * correct / total


# -----------------------------
def main(cfg):
    model_name = cfg.get("train.model_name")
    out_dir = cfg.get("train.metrics_dir", "./metrics")
    save_dir = cfg.get("train.save_dir", "./models_trained")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    init_log_files(model_name, out_dir)

    # datasets
    train_ds = MultimodalTripleDataset(
        cfg.get("dataset.root_dir"),
        cfg.get("dataset.cue_root"),
        cfg.get("dataset.lip_regions_root"),
        input_size=cfg.get("dataset.input_size"),
        split="train",
        cue_mode=cfg.get("dataset.cue_mode", "emotion"),
        embed_model=cfg.get("dataset.embed_model", "sentence-transformers/all-mpnet-base-v2"),
        cache_dir=cfg.get("dataset.cache_dir", ".cache_cues")
    )

    val_ds = MultimodalTripleDataset(
        cfg.get("dataset.root_dir"),
        cfg.get("dataset.cue_root"),
        cfg.get("dataset.lip_regions_root"),
        input_size=cfg.get("dataset.input_size"),
        split="val",
        cue_mode=cfg.get("dataset.cue_mode", "emotion"),
        embed_model=cfg.get("dataset.embed_model", "sentence-transformers/all-mpnet-base-v2"),
        cache_dir=cfg.get("dataset.cache_dir", ".cache_cues")
    )

    test_ds = MultimodalTripleDataset(
        cfg.get("dataset.root_dir"),
        cfg.get("dataset.cue_root"),
        cfg.get("dataset.lip_regions_root"),
        input_size=cfg.get("dataset.input_size"),
        split="test",
        cue_mode=cfg.get("dataset.cue_mode", "emotion"),
        embed_model=cfg.get("dataset.embed_model", "sentence-transformers/all-mpnet-base-v2"),
        cache_dir=cfg.get("dataset.cache_dir", ".cache_cues")
    )

    batch = cfg.get("train.batch", 4)
    workers = cfg.get("train.workers", 4)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate_fn_triple, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, collate_fn=collate_fn_triple, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, collate_fn=collate_fn_triple, num_workers=workers, pin_memory=True)

    # classes / dims
    num_classes = cfg.get("dataset.num_classes", None)
    if num_classes is None:
        num_classes = len(train_ds.audio_ds.classes)

    cue_dim = next(iter(train_ds.desc2vec.values())).shape[0]

    video_cfg = cfg.get("video_model_config", None)

    if model_name == "early_fusion_mobile":
        model = MultimodalAttentionEarly(num_classes=num_classes, cue_dim=cue_dim, video_cfg=video_cfg).to(device)
    elif model_name == "middle_fusion_mobile":
        model = MultimodalAttentionMiddle(num_classes=num_classes, cue_dim=cue_dim, video_cfg=video_cfg).to(device)
    elif model_name == "late_fusion_mobile":
        model = MultimodalAttentionLate(num_classes=num_classes, cue_dim=cue_dim, video_cfg=video_cfg).to(device)
    elif model_name == "early_fusion_resnet":
        model = MultimodalAttentionEarlyResNet(num_classes=num_classes, cue_dim=cue_dim, video_cfg=video_cfg).to(device)
    elif model_name == "middle_fusion_resnet":
        model = MultimodalAttentionMiddleResNet(num_classes=num_classes, cue_dim=cue_dim, video_cfg=video_cfg).to(device)
    elif model_name == "late_fusion_resnet":
        model = MultimodalAttentionLateResNet(num_classes=num_classes, cue_dim=cue_dim, video_cfg=video_cfg).to(device)
    else:
        model = MultimodalThreeNet(num_classes=num_classes, cue_dim=cue_dim,
                               audio_input_size=cfg.get("dataset.input_size"), video_cfg=video_cfg).to(device)
        raise ValueError(f"Unknown model name: {model_name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("train.lr", 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_acc = 0.0
    epochs = cfg.get("train.epochs", 30)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None, train=False)

        scheduler.step(val_loss)

        is_best = val_acc > best_val_acc
        best_val_acc = max(best_val_acc, val_acc)

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }
        torch.save(ckpt, os.path.join(save_dir, f"{model_name}_checkpoint.pth"))
        if is_best:
            torch.save(ckpt, os.path.join(save_dir, "model_best.pth"))

        test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer=None, train=False)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")

        log_to_files(model_name, out_dir, epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)

    # final best evaluation
    best_ckpt_path = os.path.join(save_dir, "model_best.pth")
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer=None, train=False)
        print(f"\nFinal Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        with open(os.path.join(out_dir, f"{model_name}_training_log.txt"), "a") as f:
            f.write(f"Final Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%\n")
    else:
        print("No best model saved; skipping final test.")

if __name__ == "__main__":
    cfg_path = "/home/aswath/Projects/capstone/multimodel_lipread/audio_cues_video/configs/acv_config.yaml"
    cfg = load_config(cfg_path)
    main(cfg)
