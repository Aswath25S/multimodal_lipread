import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class LateFusionFast(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()

        # -----------------------------
        # Audio encoder + classifier
        # -----------------------------
        in_channels = config.get("dataset.audio_channels", 1)
        self.audio_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.audio_fc = nn.Linear(16, config.get("model.audio_feature_dim", 128))
        self.audio_classifier = nn.Linear(config.get("model.audio_feature_dim", 128), num_classes)

        # -----------------------------
        # Video encoder + classifier
        # -----------------------------
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        base.classifier = nn.Identity()
        self.video_cnn = base

        self.video_lstm = nn.LSTM(
            input_size=576,           # MobileNetV3 small feature dim
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.video_classifier = nn.Linear(128 * 2, num_classes)

        # -----------------------------
        # Late fusion weight
        # -----------------------------
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, audio, video):
        # -----------------------------
        # Audio forward
        # -----------------------------
        # audio: [B, H, W] or [B, C, H, W]
        if audio.dim() == 3:
            audio = audio.unsqueeze(1)  # add channel if missing
        a = self.audio_cnn(audio).view(audio.size(0), -1)
        a = self.audio_fc(a)
        a_logits = self.audio_classifier(a)

        # -----------------------------
        # Video forward
        # -----------------------------
        B, C, T, H, W = video.shape
        v = video.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        v = v.view(B * T, C, H, W)                     # flatten for CNN
        v = self.video_cnn(v)                          # (B*T, 576)
        v = v.view(B, T, -1)                           # (B, T, 576)
        _, (h_n, _) = self.video_lstm(v)              # h_n: (num_layers*2, B, hidden)
        v = torch.cat([h_n[0], h_n[1]], dim=1)        # concat forward & backward hidden
        v_logits = self.video_classifier(v)

        # -----------------------------
        # Late fusion
        # -----------------------------
        fused = self.alpha * a_logits + (1 - self.alpha) * v_logits
        return fused

# -----------------------------
# Factory
# -----------------------------
def create_late_fusion_fast(num_classes, config):
    return LateFusionFast(num_classes, config)
