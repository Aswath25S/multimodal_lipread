import torch
import torch.nn as nn
import torchvision.models as models


# ==================================================
# TimeDistributed
# ==================================================
class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B * T, C, H, W)
        out = self.module(x)
        return out.reshape(B, T, -1)


# ==================================================
# Video Encoder: ResNet18 + BiLSTM
# ==================================================
class ResNetLSTM(nn.Module):
    def __init__(self, feature_dim=256, pretrained=True, dropout=0.3):
        super().__init__()
        base = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        base.fc = nn.Identity()

        self.cnn = base
        self.td = TimeDistributed(self.cnn)

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=feature_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.output_dim = feature_dim

    def forward(self, x):
        x = self.td(x)
        x, _ = self.lstm(x)
        return x


# ==================================================
# Attention Mechanism
# ==================================================
class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, q, k):
        Q = self.query(q).unsqueeze(1)
        K = self.key(k)
        V = self.value(k)

        attn = torch.softmax((Q @ K.transpose(-1, -2)) * self.scale, dim=-1)
        return (attn @ V).squeeze(1)


# ==================================================
# Early Fusion Model
# ==================================================
class EarlyAttentionResNet(nn.Module):
    def __init__(self, num_classes, cue_dim=768, pretrained=True):
        super().__init__()

        self.video_encoder = ResNetLSTM(pretrained=pretrained)

        self.cue_proj = nn.Sequential(
            nn.Linear(cue_dim, 256),
            nn.ReLU()
        )

        self.attn = AttentionFusion(256)

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, cue, video):
        video_feats = self.video_encoder(video)
        cue_feat = self.cue_proj(cue)
        fused = self.attn(cue_feat, video_feats)
        return self.classifier(fused)
