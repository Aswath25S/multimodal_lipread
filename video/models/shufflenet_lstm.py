import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ShuffleNet_V2_X0_5_Weights,
    ShuffleNet_V2_X1_0_Weights
)



class TimeDistributed(nn.Module):
    """
    Apply a module across time: (B, C, T, H, W) → (B, T, F)
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        out = self.module(x)
        out = out.view(B, T, -1)
        return out


class ShuffleNet2DBiLSTM(nn.Module):
    """
    Lightweight ShuffleNetV2 + BiLSTM architecture for lip reading.
    """
    def __init__(self, num_classes, config):
        super().__init__()

        # ============================================
        # 1) Select ShuffleNet version
        # ============================================
        version = config.get("model.shufflenet_version", "0.5x")  
        if version == "0.5x":
            base_model = models.shufflenet_v2_x0_5(
                weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
            )
        else:
            base_model = models.shufflenet_v2_x1_0(
                weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
            )

        # Remove classifier
        self.cnn_features = nn.Sequential(
            base_model.conv1,
            base_model.maxpool,
            base_model.stage2,
            base_model.stage3,
            base_model.stage4,
            base_model.conv5
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Determine CNN output dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 44, 44)
            f = self.cnn_features(dummy)
            f = self.global_pool(f)
            cnn_output_dim = f.view(-1).shape[0]

        # Wrap CNN in TimeDistributed
        self.time_cnn = TimeDistributed(nn.Sequential(
            self.cnn_features,
            self.global_pool,
            nn.Flatten()
        ))

        # ============================================
        # 2) BiLSTM temporal model
        # ============================================
        feature_dim = config.get("model.feature_dim", 512)
        dropout = config.get("model.dropout", 0.4)

        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=feature_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Final classification layer
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # CNN per-frame features → (B, T, F)
        x = self.time_cnn(x)

        # BiLSTM temporal modeling
        x, _ = self.lstm(x)

        # Take final timestep
        x = x[:, -1]

        x = self.relu(x)
        x = self.dropout(x)

        return self.fc(x)


def create_model(num_classes, config):
    return ShuffleNet2DBiLSTM(num_classes, config)
