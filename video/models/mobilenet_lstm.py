import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)            # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)           # (B*T, C, H, W)
        out = self.module(x)                    # (B*T, F)
        out = out.reshape(B, T, -1)             # (B, T, F)
        return out


class MobileNetLSTM(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()

        feature_dim = config.get("model.feature_dim", 256)
        dropout = config.get("model.dropout", 0.3)

        # Load MobileNetV2 backbone
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # Remove classifier → keep CNN features
        base.classifier = nn.Identity()

        # Add global average pooling
        self.cnn = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()     # output = 1280 dim
        )

        cnn_output_dim = 1280  # MobileNetV2 final feature size

        # TimeDistributed wrapper
        self.td = TimeDistributed(self.cnn)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=feature_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.td(x)          # → (B, T, 1280)
        x, _ = self.lstm(x)     # → (B, T, feature_dim)
        x = x[:, -1, :]         # last time-step
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc(x)
        return x


def create_model(num_classes, config):
    return MobileNetLSTM(num_classes, config)
