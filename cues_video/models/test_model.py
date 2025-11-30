import torch
import torch.nn as nn
import torchvision.models as models


# ----------------------------
# TimeDistributed wrapper
# ----------------------------
class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B * T, C, H, W)
        out = self.module(x)
        out = out.reshape(B, T, -1)
        return out


# ----------------------------
# Video Encoder: MobileNet + BiLSTM
# ----------------------------
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
        return x[:, -1, :]


# ----------------------------
# Cue Encoder
# ----------------------------
class CueEncoder(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256)
        )
        self.output_dim = 256

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Multimodal Model
# ----------------------------
class MultimodalCueVideoNet(nn.Module):
    def __init__(self, num_classes, cue_dim=768, pretrained=True):
        super().__init__()

        self.video_encoder = MobileNetLSTM(pretrained=pretrained)
        self.cue_encoder = CueEncoder(input_dim=cue_dim)

        fused_dim = self.video_encoder.output_dim + self.cue_encoder.output_dim

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, cue, video):
        v = self.video_encoder(video)
        c = self.cue_encoder(cue)

        x = torch.cat([v, c], dim=1)
        x = self.fusion(x)
        return self.classifier(x)
