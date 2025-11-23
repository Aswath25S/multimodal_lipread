import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ============================================================
# AUDIO ENCODER
# ============================================================

class AudioEncoderFast(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.get("dataset.audio_channels", 1)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, config.get("model.audio_feature_dim", 128))
        self.output_dim = config.get("model.audio_feature_dim", 128)

    def forward(self, x):
        # x: [B, H, W] or [B, 1, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(1)  # add channel dim if missing
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ============================================================
# VIDEO ENCODER
# ============================================================

class VideoEncoderFast(nn.Module):
    def __init__(self, config):
        super().__init__()
        lstm_hidden = config.get("video.lstm_hidden", 128)

        # MobileNetV3 small backbone
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        base.classifier = nn.Identity()
        self.cnn = base

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=576,  # MobileNet small final feature dim
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.output_dim = lstm_hidden * 2

    def forward(self, x):
        """
        x: (B, 3, T, H, W)
        """
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        x = x.view(B * T, C, H, W)               # flatten for CNN
        feats = self.cnn(x)                      # (B*T, 576)
        feats = feats.view(B, T, -1)             # (B, T, 576)
        _, (h_n, _) = self.lstm(feats)           # h_n: (num_layers*2, B, hidden_size)
        # concat forward & backward hidden
        feats = torch.cat([h_n[0], h_n[1]], dim=1)  # (B, hidden*2)
        return feats

# ============================================================
# EARLY FUSION MODEL
# ============================================================

class EarlyFusionFast(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        self.audio_encoder = AudioEncoderFast(config)
        self.video_encoder = VideoEncoderFast(config)

        fusion_dim = self.audio_encoder.output_dim + self.video_encoder.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, audio, video):
        a = self.audio_encoder(audio)
        v = self.video_encoder(video)
        fused = torch.cat([a, v], dim=1)
        return self.classifier(fused)

# ============================================================
# FACTORY
# ============================================================

def create_early_fusion_fast(num_classes, config):
    return EarlyFusionFast(num_classes, config)
