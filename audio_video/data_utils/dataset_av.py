import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append('/home/aswath/Projects/capstone/multimodel_lipread/audio_video')

from utils.audio_processor import AudioProcessor
from utils.video_processor import VisualDataset


class GLipsMultimodalDataset(Dataset):
    """
    Dataset returning both audio and video features for the same sample.
    """

    def __init__(self, root_dir, input_size_audio, split='train', transform_audio=None, transform_video=None):
        self.root_dir = root_dir
        self.input_size_audio = input_size_audio
        self.split = split
        self.transform_audio = transform_audio
        self.transform_video = transform_video

        self.audio_processor = AudioProcessor()
        self.video_dataset = VisualDataset(
            root_dir=root_dir,
            lip_regions_dir=root_dir + "_lip_regions",
            split=split,
            transform=transform_video
        )

        # Ensure audio samples exist
        self.samples = []
        for sample in self.video_dataset.samples:
            video_path, label = sample
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            audio_path = os.path.join(root_dir, "lipread_files",
                                      video_path.split(os.sep)[-3],
                                      split,
                                      base_name + ".m4a")
            if os.path.exists(audio_path):
                self.samples.append({
                    "audio_path": audio_path,
                    "video_path": video_path,
                    "label": label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
    
        # Process audio
        mel_spec = self.audio_processor.process_audio_file(sample['audio_path'])
        mel_spec = self.audio_processor.normalize_spectrogram(mel_spec)
        if self.transform_audio:
            mel_spec = self.transform_audio(mel_spec)
        mel_spec = mel_spec[:80, :self.input_size_audio]
        if isinstance(mel_spec, torch.Tensor):
            mel_spec = mel_spec.float()
        else:
            mel_spec = torch.tensor(mel_spec, dtype=torch.float32)


        # Process video
        lip_regions = np.load(sample['video_path']).astype(np.float32) / 255.0
        lip_regions = torch.tensor(lip_regions).permute(3, 0, 1, 2)  # C, T, H, W
        if self.transform_video:
            lip_regions = self.transform_video(lip_regions)

        label = torch.tensor(sample["label"], dtype=torch.long)

        return mel_spec, lip_regions, label
