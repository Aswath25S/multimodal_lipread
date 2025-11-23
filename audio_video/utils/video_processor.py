"""
Data loader for GLips AVSR project.
Handles loading and preprocessing of visual data.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

# Add parent directory to path to import config
sys.path.append('/home/aswath/Projects/capstone/multimodel_lipread/video')
from config.config import load_config


class VisualDataset(Dataset):
    """Dataset for visual speech recognition using preprocessed lip regions."""

    def __init__(self, root_dir, lip_regions_dir, split='train', transform=None):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Path to GLips_4
            lip_regions_dir (str): Path to GLips_4_lip_regions
            split (str): One of ['train', 'val', 'test']
        """
        self.root_dir = root_dir
        self.lip_regions_dir = lip_regions_dir
        self.split = split
        self.transform = transform

        # Classes found inside: root_dir/lipread_files/
        self.class_dir = os.path.join(root_dir, "lipread_files")
        self.lip_regions_class_dir = os.path.join(lip_regions_dir, "lipread_files")

        # class list
        self.classes = sorted([
            d for d in os.listdir(self.class_dir)
            if os.path.isdir(os.path.join(self.class_dir, d))
        ])

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Build sample list
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []

        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]

            # Example path: GLips_4/lipread_files/aufgaben/train/
            video_split_dir = os.path.join(
                self.class_dir, class_name, self.split
            )

            if not os.path.exists(video_split_dir):
                continue

            video_files = [
                f for f in os.listdir(video_split_dir) if f.endswith(".mp4")
            ]

            for file_name in video_files:
                base = os.path.splitext(file_name)[0]

                # Example: GLips_4_lip_regions/lipread_files/aufgaben/train/aufgaben_001.npy
                lip_region_path = os.path.join(
                    self.lip_regions_class_dir,
                    class_name,
                    self.split,
                    base + ".npy"
                )

                if os.path.exists(lip_region_path):
                    samples.append((lip_region_path, class_idx))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lip_regions_path, label = self.samples[idx]

        lip_regions = np.load(lip_regions_path).astype(np.float32) / 255.0

        if self.transform:
            lip_regions = self.transform(lip_regions)

        # (T, H, W, C) → (C, T, H, W)
        lip_regions = torch.tensor(lip_regions).permute(3, 0, 1, 2)

        return {
            "lip_regions": lip_regions,
            "label": torch.tensor(label, dtype=torch.long)
        }


class MixupTransform:
    """Mixup augmentation for sequences of lip images."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch):
        x, y = batch["lip_regions"], batch["label"]

        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1

        batch_size = x.size(0)
        indices = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[indices]

        return {
            "lip_regions": mixed_x,
            "label_a": y,
            "label_b": y[indices],
            "lam": lam
        }


def get_data_loaders(config_path):
    """
    Create DataLoaders for train, val, and test splits.
    """

    # Load config
    config = load_config(config_path)

    dataset_path = config.get('dataset.root_dir')

    # Path to preprocessed lip region .npy files
    lip_regions_dir = os.path.join(
        os.path.dirname(dataset_path),
        os.path.basename(dataset_path) + "_lip_regions"
    )

    if not os.path.exists(lip_regions_dir):
        raise FileNotFoundError(
            f"Preprocessed lip regions not found at {lip_regions_dir}. "
            f"Run visual_preprocessing.py first."
        )

    batch_size = config.get("training.batch_size", 4)

    # Create datasets that load based on folder structure
    train_dataset = VisualDataset(dataset_path, lip_regions_dir, split="train")
    val_dataset = VisualDataset(dataset_path, lip_regions_dir, split="val")
    test_dataset = VisualDataset(dataset_path, lip_regions_dir, split="test")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
