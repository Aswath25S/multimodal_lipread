import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights


class TimeDistributed(nn.Module):
    """
    Applies a module to every time slice: (B, C, T, H, W) -> (B, T, F)
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        out = self.module(x)
        out = out.reshape(B, T, -1)
        return out


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for Transformers.
    """

    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)

        div = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, F)
        T = x.size(1)
        return x + self.pe[:, :T, :].to(x.device)


class ResNet2DTransformer(nn.Module):
    """
    ResNet backbone + Temporal TransformerEncoder.
    """
    def __init__(self, num_classes, config):
        super().__init__()

        # ===============================
        # Load ResNet Backbone
        # ===============================
        version = config.get("model.resnet_version", 18)
        if version == 18:
            base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            base_model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        # Keep 3-channel input
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove classifier
        self.cnn_features = nn.Sequential(*list(base_model.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Determine CNN output dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 44, 44)
            f = self.cnn_features(dummy)
            f = self.global_pool(f)
            cnn_out_dim = f.view(-1).size(0)

        # TimeDistributed CNN
        self.time_cnn = TimeDistributed(nn.Sequential(
            self.cnn_features,
            self.global_pool,
            nn.Flatten()
        ))

        # ===============================
        # Transformer Encoder
        # ===============================
        transformer_dim = config.get("model.transformer_dim", 256)
        num_layers = config.get("model.num_layers", 2)
        num_heads = config.get("model.num_heads", 4)
        dropout = config.get("model.dropout", 0.2)

        # Project CNN features → Transformer dimension
        self.proj_in = nn.Linear(cnn_out_dim, transformer_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(transformer_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=transformer_dim * 4  # transformer rule-of-thumb
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Final classifier
        self.fc = nn.Linear(transformer_dim, num_classes)

    def forward(self, x):
        # CNN per frame → (B, T, F)
        x = self.time_cnn(x)

        # Project to transformer dimension
        x = self.proj_in(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer
        x = self.transformer(x)

        # Temporal aggregation (mean pooling)
        x = x.mean(dim=1)

        x = self.relu(x)
        x = self.dropout(x)
        return self.fc(x)


def create_model(num_classes, config):
    return ResNet2DTransformer(num_classes, config)
