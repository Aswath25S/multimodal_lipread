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
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        out = self.module(x)
        out = out.view(B, T, -1)
        return out


class TemporalAttention(nn.Module):
    """
    Multi-head attention over time based on frame embeddings.
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        # Self-attention over time dimension
        # x = (B, T, F)
        attn_out, _ = self.attn(x, x, x)
        return attn_out


class ResNet2DAttention(nn.Module):
    """
    ResNet backbone + Multi-Head Attention across time.
    """
    def __init__(self, num_classes, config):
        super().__init__()

        # =======================================================
        # Load ResNet backbone (18 or 34)
        # =======================================================
        version = config.get("model.resnet_version", 18)
        if version == 18:
            base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            base_model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        # Modify first conv if needed (still 3-channel)
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the classifier
        self.cnn_features = nn.Sequential(*list(base_model.children())[:-2])

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Determine output dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 44, 44)
            f = self.cnn_features(dummy)
            f = self.global_pool(f)
            cnn_output_dim = f.view(-1).shape[0]

        # TimeDistributed CNN
        self.time_cnn = TimeDistributed(nn.Sequential(
            self.cnn_features,
            self.global_pool,
            nn.Flatten()
        ))

        # =======================================================
        # Temporal Multi-Head Attention
        # =======================================================
        attn_dim = config.get("model.attention_dim", cnn_output_dim)
        num_heads = config.get("model.num_heads", 4)
        dropout = config.get("model.dropout", 0.3)

        # Project CNN embeddings → attention dimension
        self.proj_in = nn.Linear(cnn_output_dim, attn_dim)

        # Multi-head self-attention across time
        self.attention = TemporalAttention(attn_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Final class prediction
        self.fc = nn.Linear(attn_dim, num_classes)

    def forward(self, x):
        # CNN on each frame → (B, T, F)
        x = self.time_cnn(x)

        # Project to attention dimension
        x = self.proj_in(x)

        # Temporal attention
        x = self.attention(x)

        # Use the final attended vector (mean also works)
        x = x.mean(dim=1)

        x = self.relu(x)
        x = self.dropout(x)

        return self.fc(x)


def create_model(num_classes, config):
    return ResNet2DAttention(num_classes, config)
