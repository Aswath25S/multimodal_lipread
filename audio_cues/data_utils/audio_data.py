import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys

from utils.audio_processor import AudioProcessor

class GLipsDataset(Dataset):
    def __init__(self, root_dir, input_size, split='train', transform=None):
        """
        Args:
            root_dir (str): Path to GLips_40 dataset
            split (str): One of 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on spectrogram
        """
        self.root_dir = root_dir
        self.input_size = input_size
        self.class_dir = os.path.join(root_dir, 'lipread_files')
        self.split = split
        self.transform = transform
        self.audio_processor = AudioProcessor()

        self.classes = sorted([word.name for word in os.scandir(self.class_dir) if word.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []

        for word in self.classes:
            word_dir = os.path.join(self.class_dir, word, self.split)
            if os.path.exists(word_dir):
                audio_files = [f for f in os.listdir(word_dir) if f.endswith('.m4a')]
                for audio_file in audio_files:
                    self.samples.append({
                        'audio_path': os.path.join(word_dir, audio_file),
                        'label': self.class_to_idx[word]
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and process audio
        mel_spec = self.audio_processor.process_audio_file(sample['audio_path'])
        mel_spec = self.audio_processor.normalize_spectrogram(mel_spec)
        
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        return mel_spec[:80, :self.input_size], torch.tensor(sample['label'], dtype=torch.long)

if __name__ == "__main__":
    directory_path = "/home/aswath/Projects/capstone/multimodel_lipread/data/GLips_4"
    dataset = GLipsDataset(directory_path, 117)
    spec, label = dataset[10]
    print(spec.shape)