"""
Data loader for GLips AVSR project.
Handles loading and preprocessing of visual data.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import sys
import random

# Add parent directory to path to import config
sys.path.append('/home/aswath/Projects/capstone/multimodel_lipread/video')
from config.config import load_config

class VisualDataset(Dataset):
    """Dataset for visual speech recognition using preprocessed lip regions."""
    
    def __init__(self, root_dir, lip_regions_dir, split='train', transform=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Path to the original GLips dataset
            lip_regions_dir (str): Path to the preprocessed lip regions
            split (str): Data split ('train', 'val', 'test')
            transform (callable, optional): Optional transform to be applied to samples
        """
        self.root_dir = root_dir
        self.lip_regions_dir = lip_regions_dir
        self.split = split
        self.transform = transform
        
        # Get all class directories
        self.classes = [d for d in os.listdir(root_dir) 
                         if os.path.isdir(os.path.join(root_dir, d))]
        self.classes.sort()  # Ensure consistent ordering
        
        # Create class to index mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Build samples list
        self.samples = self._build_samples()
    
    def _build_samples(self):
        """
        Build a list of samples (file paths and labels).
        
        Returns:
            list: List of (file_path, label) pairs
        """
        samples = []
        
        # Determine split boundaries
        train_end = 0.7  # 70% for training
        val_end = 0.85   # Next 15% for validation
        
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Get video files for this class
            video_files = [f for f in os.listdir(class_dir) 
                           if f.endswith('.mp4')]
            video_files.sort()  # Ensure consistent ordering
            
            # Determine split indices
            n_samples = len(video_files)
            train_idx = int(n_samples * train_end)
            val_idx = int(n_samples * val_end)
            
            # Select files based on split
            if self.split == 'train':
                files = video_files[:train_idx]
            elif self.split == 'val':
                files = video_files[train_idx:val_idx]
            else:  # test
                files = video_files[val_idx:]
            
            # Add samples
            for file_name in files:
                # Original video path
                video_path = os.path.join(class_name, file_name)
                
                # Corresponding lip regions path (.npy file)
                lip_regions_path = os.path.join(
                    self.lip_regions_dir, 
                    class_name, 
                    os.path.splitext(file_name)[0] + '.npy'
                )
                
                # Check if the preprocessed file exists
                if os.path.exists(lip_regions_path):
                    samples.append((lip_regions_path, class_idx))
        
        return samples
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Sample dictionary containing lip regions and label
        """
        lip_regions_path, label = self.samples[idx]
        
        # Load lip regions
        lip_regions = np.load(lip_regions_path)
        
        # Convert to tensor and normalize to [0, 1]
        lip_regions = lip_regions.astype(np.float32) / 255.0
        
        # Apply transforms if specified
        if self.transform:
            lip_regions = self.transform(lip_regions)
        
        # Convert to torch tensor and transpose to (C, T, H, W)
        # Original: (T, H, W, C) -> (C, T, H, W)
        lip_regions = torch.tensor(lip_regions).permute(3, 0, 1, 2)
        
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return {'lip_regions': lip_regions, 'label': label}


class MixupTransform:
    """
    Mixup data augmentation for sequences of images.
    
    Reference:
        Zhang et al. "mixup: Beyond Empirical Risk Minimization"
        https://arxiv.org/abs/1710.09412
    """
    
    def __init__(self, alpha=1.0):
        """
        Initialize Mixup transform.
        
        Args:
            alpha (float): Alpha parameter for Beta distribution
        """
        self.alpha = alpha
    
    def __call__(self, batch):
        """
        Apply mixup to a batch of data.
        
        Args:
            batch (dict): Batch dictionary with 'lip_regions' and 'label'
            
        Returns:
            tuple: Mixed data and mixed labels with mixing coefficient
        """
        # Unpack batch
        x, y = batch['lip_regions'], batch['label']
        
        # Generate mixing coefficient
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        # Get batch size
        batch_size = x.size(0)
        
        # Generate random indices
        indices = torch.randperm(batch_size)
        
        # Mix data
        mixed_x = lam * x + (1 - lam) * x[indices]
        
        # Return mixed data and both labels with mixing coefficient
        return {
            'lip_regions': mixed_x, 
            'label_a': y, 
            'label_b': y[indices], 
            'lam': lam
        }


def get_data_loaders(config_path):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        tuple: Training, validation, and test data loaders
    """
    # Load configuration
    config = load_config(config_path)
    
    # Get dataset paths
    dataset_path = config.get('dataset.root_dir')
    # Determine lip regions directory
    lip_regions_dir = os.path.join(
        os.path.dirname(dataset_path), 
        os.path.basename(dataset_path) + '_lip_regions'
    )
    
    # Check if preprocessed data exists
    if not os.path.exists(lip_regions_dir):
        raise FileNotFoundError(
            f"Preprocessed lip regions not found at {lip_regions_dir}. "
            f"Please run visual_preprocessing.py first."
        )
    
    # Get batch size
    batch_size = config.get('training.batch_size', 4)
    
    # Create datasets
    train_dataset = VisualDataset(dataset_path, lip_regions_dir, split='train')
    val_dataset = VisualDataset(dataset_path, lip_regions_dir, split='val')
    test_dataset = VisualDataset(dataset_path, lip_regions_dir, split='test')
    
    # Create data loaders
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
