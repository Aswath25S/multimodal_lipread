"""
Early Fusion Audio+Video Model for AVSR with MobileNet for Video.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ============================================================
# AUDIO ENCODER  (Simple CNN → features)
# ============================================================

class AudioEncoder(nn.Module):
    def __init__(self, config):
        super(AudioEncoder, self).__init__()

        in_channels = config.get("dataset.audio_channels", 1)
        feature_dim = config.get("model.audio_feature_dim", 256)

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

        self.fc = nn.Linear(128, feature_dim)
        self.output_dim = feature_dim

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ============================================================
# VIDEO ENCODER (MobileNet + BiLSTM)
# ============================================================

class VideoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        lstm_hidden = config.get("video.lstm_hidden", 256)

        # MobileNetV3 small backbone
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        base.classifier = nn.Identity()  # remove final classifier
        self.cnn = base

        self.lstm = nn.LSTM(
            input_size=576,  # MobileNet small final feature dim
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
        feats = self.cnn(x)  # (B*T, 576)
        feats = feats.view(B, T, -1)
        feats, _ = self.lstm(feats)
        feats = feats[:, -1]  # last timestep
        return feats

# ============================================================
# EARLY FUSION MODEL
# ============================================================

class EarlyFusionAVMobileNet(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        self.audio_encoder = AudioEncoder(config)
        self.video_encoder = VideoEncoder(config)

        fusion_dim = self.audio_encoder.output_dim + self.video_encoder.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, audio, video):
        audio = audio.unsqueeze(1)  # ensure channel dim
        a = self.audio_encoder(audio)
        v = self.video_encoder(video)
        fused = torch.cat([a, v], dim=1)
        return self.classifier(fused)

# ============================================================
# FACTORY
# ============================================================

def create_early_fusion_mobilenet_model(num_classes, config):
    return EarlyFusionAVMobileNet(num_classes, config)
