# multimodal/train_mm.py
import torch, yaml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_utils.dataset import MultimodalDataset, collate_fn
from models.test_model import MultimodalNet

from configs.config import load_config


def run_epoch(model, loader, opt=None, criterion=None, train=True):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0

    for mel, cue, label in loader:
        mel, cue, label = mel.to(device), cue.to(device), label.to(device)

        out = model(mel, cue)
        loss = criterion(out, label)

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        pred = out.argmax(1)
        total += label.size(0)
        correct += pred.eq(label).sum().item()
        loss_sum += loss.item()

    return loss_sum / len(loader), 100 * correct / total


def run_epoch(model, loader, opt=None, criterion=None, train=True):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0

    for mel, cue, label in loader:
        mel, cue, label = mel.to(device), cue.to(device), label.to(device)

        out = model(mel, cue)
        loss = criterion(out, label)

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        pred = out.argmax(1)
        total += label.size(0)
        correct += pred.eq(label).sum().item()
        loss_sum += loss.item()

    return loss_sum / len(loader), 100 * correct / total


device = "cuda" if torch.cuda.is_available() else "cpu"


def main(cfg):
    train_ds = MultimodalDataset(root_dir=cfg.get("dataset.root_dir"), cue_root=cfg.get("dataset.cue_root"), input_size=cfg.get("dataset.input_size"), split="train")
    val_ds   = MultimodalDataset(root_dir=cfg.get("dataset.root_dir"), cue_root=cfg.get("dataset.cue_root"), input_size=cfg.get("dataset.input_size"), split="val")
    test_ds  = MultimodalDataset(root_dir=cfg.get("dataset.root_dir"), cue_root=cfg.get("dataset.cue_root"), input_size=cfg.get("dataset.input_size"), split="test")

    train_loader = DataLoader(train_ds, batch_size=cfg.get("train.batch"), shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=cfg.get("train.batch"), shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=cfg.get("train.batch"), shuffle=False, collate_fn=collate_fn)

    # text_dim = next(iter(train_ds.desc2vec.values())).shape[0]
    # model = MultimodalNet(num_classes=cfg.get("dataset.num_classes"), cue_dim=text_dim).to(device)
    model = MultimodalNet(
        num_classes=cfg.get("dataset.num_classes"),
        cue_dim=768
    ).to(device)


    opt = optim.Adam(model.parameters(), lr=cfg.get("train.lr"))
    criterion = nn.CrossEntropyLoss()

    best = 0
    for epoch in range(cfg.get("train.epochs")):
        tr = run_epoch(model, train_loader, opt, criterion, True)
        va = run_epoch(model, val_loader, None, criterion, False)

        print(f"Epoch {epoch+1}")
        print(f" Train: {tr}")
        print(f" Val:   {va}")

        if va[1] > best:
            best = va[1]
            torch.save(model.state_dict(), "best_multimodal.pth")

    model.load_state_dict(torch.load("best_multimodal.pth"))
    test = run_epoch(model, test_loader, None, criterion, False)
    print("Final Test:", test)


if __name__ == "__main__":
    cfg = load_config("/home/aswath/Projects/capstone/multimodel_lipread/audio_cues/configs/ac_config.yaml")
    main(cfg)
