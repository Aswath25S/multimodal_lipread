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
# Late Fusion Model
# --------------------------------------------------
class LateAttentionFusion(nn.Module):
    def __init__(self, num_classes, cue_dim=768, pretrained=True):
        super().__init__()

        self.video_encoder = MobileNetLSTM(pretrained=pretrained)
        self.video_head = nn.Linear(256, num_classes)

        self.cue_head = nn.Sequential(
            nn.Linear(cue_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self.attn = nn.Sequential(
            nn.Linear(num_classes * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, cue, video):
        v_feat = self.video_encoder(video)[:, -1, :]
        v_logits = self.video_head(v_feat)
        c_logits = self.cue_head(cue)

        weights = self.attn(torch.cat([v_logits, c_logits], dim=1))
        wv, wc = weights[:, 0:1], weights[:, 1:2]

        return wv * v_logits + wc * c_logits
