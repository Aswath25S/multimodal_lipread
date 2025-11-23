import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNOnly(nn.Module):
    """
    CNN-only model for visual speech recognition.
    Uses 2D CNN to extract per-frame features and 1D convolutions over time
    to capture temporal patterns.
    """
    def __init__(self, num_classes, config):
        super().__init__()

        # Per-frame 2D CNN (lightweight)
        self.frame_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 44x44 -> 22x22

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 22x22 -> 11x11

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # output: 128-dim vector
        )

        # Temporal convolution over sequence of frames
        temporal_channels = config.get("model.temporal_channels", 128)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(128, temporal_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(temporal_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(temporal_channels, temporal_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(temporal_channels),
            nn.ReLU(inplace=True)
        )

        # Dropout before final classification
        dropout = config.get("model.dropout", 0.3)
        self.dropout = nn.Dropout(dropout)

        # Fully connected classifier
        self.fc = nn.Linear(temporal_channels, num_classes)

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape

        # Process each frame with 2D CNN
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)
        x = self.frame_cnn(x)          # (B*T, 128, 1, 1)
        x = x.view(B, T, -1)           # (B, T, 128)

        # Temporal conv expects (B, C, T)
        x = x.permute(0, 2, 1)         # (B, 128, T)
        x = self.temporal_conv(x)      # (B, temporal_channels, T)

        # Temporal pooling: mean over time
        x = x.mean(dim=2)              # (B, temporal_channels)

        x = self.dropout(x)
        return self.fc(x)


def create_model(num_classes, config):
    return CNNOnly(num_classes, config)
