import torch
import torch.nn as nn


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)        # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)       # (B*T, C, H, W)
        out = self.module(x)                # (B*T, F)
        return out.reshape(B, T, -1)        # (B, T, F)


class VGGLite(nn.Module):
    """
    A small custom VGG suitable for small GPUs.
    Not nearly as heavy as VGG16.
    """
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),      # 44 → 22

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),      # 22 → 11

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1))    # → 128 features
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)        # flatten to (128,)


class VGGLSTM(nn.Module):
    """
    VGG-style CNN + BiLSTM for sequence classification.
    Designed to be lightweight for 2GB GPUs.
    """
    def __init__(self, num_classes, config):
        super().__init__()

        feature_dim = config.get("model.feature_dim", 256)
        dropout = config.get("model.dropout", 0.5)

        self.td = TimeDistributed(VGGLite())

        cnn_output_dim = 128   # final feature size from VGGLite

        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=feature_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        x = self.td(x)          # (B, T, 128)
        x, _ = self.lstm(x)     # (B, T, feature_dim)
        x = x[:, -1, :]         # last time step
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def create_model(num_classes, config):
    return VGGLSTM(num_classes, config)
