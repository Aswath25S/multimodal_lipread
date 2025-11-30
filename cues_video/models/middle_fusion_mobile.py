import torch
import torch.nn as nn
import torchvision.models as models


# --------------------------------------------------
# TimeDistributed
# --------------------------------------------------
class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B * T, C, H, W)
        out = self.module(x)
        out = out.reshape(B, T, -1)
        return out


# --------------------------------------------------
# Video Encoder
# --------------------------------------------------
class MobileNetLSTM(nn.Module):
    def __init__(self, feature_dim=256, dropout=0.3, pretrained=True):
        super().__init__()

        base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        base.classifier = nn.Identity()

        self.cnn = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.td = TimeDistributed(self.cnn)

        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=feature_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        self.output_dim = feature_dim

    def forward(self, x):
        x = self.td(x)
        x, _ = self.lstm(x)
        return x


# --------------------------------------------------
# Attention
# --------------------------------------------------
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


# --------------------------------------------------
# Middle Fusion Model
# --------------------------------------------------
class MiddleAttentionFusion(nn.Module):
    def __init__(self, num_classes, cue_dim=768, pretrained=True):
        super().__init__()

        self.video_encoder = MobileNetLSTM(pretrained=pretrained)

        self.cue_encoder = nn.Sequential(
            nn.Linear(cue_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.attn = AttentionFusion(256)

        self.fusion = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, cue, video):
        video_feats = self.video_encoder(video)
        video_last = video_feats[:, -1, :]

        cue_feat = self.cue_encoder(cue)

        attended = self.attn(cue_feat, video_feats)
        fused = torch.cat([video_last, attended], dim=1)

        return self.classifier(self.fusion(fused))
