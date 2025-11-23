import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

"""
Late-Level Fusion Audio+Video Model for AVSR with MobileNet.
Audio and video produce separate logits, then fused before final prediction.
"""

# ============================================================
# Audio Encoder
# ============================================================
class AudioEncoderLate(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.get("dataset.audio_channels", 1)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, config.get("model.audio_feature_dim", 256))
        self.output_dim = config.get("model.audio_feature_dim", 256)

    def forward(self, x):
        # Handle both [B,H,W] and [B,C,H,W]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ============================================================
# Video Encoder
# ============================================================
class VideoEncoderLate(nn.Module):
    def __init__(self, config):
        super().__init__()
        lstm_hidden = config.get("video.lstm_hidden", 256)
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        base.classifier = nn.Identity()
        self.cnn = base
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
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        x = x.view(B * T, C, H, W)                 # flatten for CNN
        feats = self.cnn(x)                        # (B*T, 576)
        feats = feats.view(B, T, -1)               # (B, T, 576)
        _, (h_n, _) = self.lstm(feats)            # h_n: (num_layers*2, B, hidden)
        feats = torch.cat([h_n[0], h_n[1]], dim=1)  # concat forward & backward hidden
        return feats

# ============================================================
# Late Fusion Model
# ============================================================
class LateFusionAVMobileNet(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        self.audio_encoder = AudioEncoderLate(config)
        self.video_encoder = VideoEncoderLate(config)

        self.audio_classifier = nn.Linear(self.audio_encoder.output_dim, num_classes)
        self.video_classifier = nn.Linear(self.video_encoder.output_dim, num_classes)

        # Optional learnable fusion weight
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, audio, video):
        a_feat = self.audio_encoder(audio)
        v_feat = self.video_encoder(video)

        a_logits = self.audio_classifier(a_feat)
        v_logits = self.video_classifier(v_feat)

        # Weighted sum fusion
        fused = self.alpha * a_logits + (1 - self.alpha) * v_logits
        return fused

# ============================================================
# Factory
# ============================================================
def create_late_fusion_mobilenet_model(num_classes, config):
    return LateFusionAVMobileNet(num_classes, config)
