"""
Early Fusion Audio+Video Model for AVSR.
Supports 4-channel audio input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# AUDIO ENCODER  (4-channel CNN → features)
# ============================================================

class AudioEncoder(nn.Module):
    def __init__(self, config):
        super(AudioEncoder, self).__init__()

        in_channels = config.get("dataset.audio_channels", 1)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(128, config.get("model.audio_feature_dim", 256))

        # REQUIRED for fusion model
        self.output_dim = config.get("model.audio_feature_dim", 256)

    def forward(self, x):
        # audio: (B, 1, 80, 117)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================
# VIDEO ENCODER (ResNet18 + BiLSTM)
# ============================================================

from torchvision.models import resnet18, ResNet18_Weights

class VideoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        lstm_hidden = config.get("video.lstm_hidden", 256)

        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        base.fc = nn.Identity()
        self.cnn = base

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.output_dim = lstm_hidden * 2

    def forward(self, x):
        """
        x: (B, 3, T, H, W)
        """
        B, C, T, H, W = x.shape

        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B,T,C,H,W)
        x = x.view(B*T, C, H, W)

        feats = self.cnn(x)          # (B*T,512)

        feats = feats.view(B, T, -1)

        feats, _ = self.lstm(feats)
        feats = feats[:, -1]         # last timestep

        return feats                 # (B, video_dim)


# ============================================================
# EARLY FUSION MODEL
# ============================================================

class EarlyFusionAV(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()

        self.audio_encoder = AudioEncoder(config)
        self.video_encoder = VideoEncoder(config)

        fusion_dim = (
            self.audio_encoder.output_dim +
            self.video_encoder.output_dim
        )

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


# ============================================================
# FACTORY
# ============================================================

def create_early_fusion_model(num_classes, config):
    return EarlyFusionAV(num_classes, config)
