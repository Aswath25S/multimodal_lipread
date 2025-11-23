"""
Middle-Level Fusion Audio+Video Model for AVSR with MobileNet.
Fusion happens after partial encoding of audio and video features.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ----------------------
# Audio Encoder (partial)
# ----------------------
class AudioEncoderMid(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.get("dataset.audio_channels", 1)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.output_dim = 64 * 20 * 29  # intermediate flattened feature size

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return x

# ----------------------
# Video Encoder (partial)
# ----------------------
class VideoEncoderMid(nn.Module):
    def __init__(self, config):
        super().__init__()
        lstm_hidden = config.get("video.lstm_hidden", 256)
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        base.classifier = nn.Identity()
        self.cnn = base

        # Optional LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=576,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.output_dim = lstm_hidden * 2

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)
        feats = feats.view(B, T, -1)
        feats, _ = self.lstm(feats)
        feats = feats[:, -1]
        return feats

# ----------------------
# Middle-Level Fusion Model
# ----------------------
class MidFusionAVMobileNet(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        self.audio_encoder = AudioEncoderMid(config)
        self.video_encoder = VideoEncoderMid(config)

        fusion_dim = self.audio_encoder.output_dim + self.video_encoder.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, audio, video):
        audio = audio.unsqueeze(1)
        a = self.audio_encoder(audio)
        v = self.video_encoder(video)
        fused = torch.cat([a, v], dim=1)
        return self.classifier(fused)

def create_mid_fusion_mobilenet_model(num_classes, config):
    return MidFusionAVMobileNet(num_classes, config)
