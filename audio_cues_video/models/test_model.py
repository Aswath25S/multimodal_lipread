# models/multimodal_three.py
import torch
import torch.nn as nn
import torchvision.models as models

# ----------------------------
# TimeDistributed wrapper (same semantics as your video model)
# ----------------------------
class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)          # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)         # (B*T, C, H, W)
        out = self.module(x)                  # (B*T, F)
        out = out.reshape(B, T, -1)           # (B, T, F)
        return out


# ----------------------------
# MobileNet + BiLSTM (inlined from your video model)
# ----------------------------
class MobileNetLSTM_Inlined(nn.Module):
    def __init__(self, num_classes, feature_dim=256, dropout=0.3, pretrained=True):
        super().__init__()

        # MobileNetV2 backbone
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        base.classifier = nn.Identity()

        self.cnn = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()   # -> 1280
        )
        cnn_output_dim = 1280

        # TimeDistributed wrapper around CNN
        self.td = TimeDistributed(self.cnn)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=feature_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.feature_dim = feature_dim
        # final classifier is not used by wrapper; multimodal model will use features directly
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.td(x)          # (B, T, 1280)
        x, _ = self.lstm(x)     # (B, T, feature_dim)
        x = x[:, -1, :]         # last time-step
        x = self.relu(x)
        x = self.drop(x)
        out = self.fc(x)
        return out

    def features(self, x):
        """
        Return the penultimate feature vector (pre-classifier) for fusion:
        (B, feature_dim)
        """
        x = self.td(x)
        x, _ = self.lstm(x)
        feat = x[:, -1, :]  # (B, feature_dim)
        return feat


# ----------------------------
# Audio encoder (ResNet on mel) - reuse your audio model style
# ----------------------------
class AudioEncoder(nn.Module):
    def __init__(self, input_size=117, pretrained=True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.fc = nn.Identity()
        self.encoder = resnet
        self.output_dim = 512

    def forward(self, x):
        # x: (B, 80, W)
        x = x.unsqueeze(1)   # (B,1,80,W)
        return self.encoder(x)  # (B,512)


# ----------------------------
# Cue encoder (MLP)
# ----------------------------
class CueEncoder(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.output_dim = 256

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Multimodal fusion net (audio + cue + video)
# ----------------------------
class MultimodalThreeNet(nn.Module):
    def __init__(self, num_classes, cue_dim=768, audio_input_size=117, video_cfg=None, pretrained=True):
        """
        video_cfg: dict-like with keys 'feature_dim' and 'dropout' to mirror your video model config.
        """
        super().__init__()
        self.audio_encoder = AudioEncoder(input_size=audio_input_size, pretrained=pretrained)
        self.cue_encoder = CueEncoder(input_dim=cue_dim)

        v_feature_dim = 256
        v_dropout = 0.3
        if video_cfg is not None:
            try:
                v_feature_dim = int(video_cfg.get("model", {}).get("feature_dim", v_feature_dim))
                v_dropout = float(video_cfg.get("model", {}).get("dropout", v_dropout))
            except Exception:
                pass

        self.video_net = MobileNetLSTM_Inlined(num_classes=num_classes, feature_dim=v_feature_dim, dropout=v_dropout, pretrained=pretrained)
        audio_dim = self.audio_encoder.output_dim
        cue_dim_out = self.cue_encoder.output_dim
        video_dim = self.video_net.feature_dim

        fusion_dim = audio_dim + cue_dim_out + video_dim

        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU()
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, mel, cue, lip_regions):
        """
        mel: (B,80,W)
        cue: (B, cue_dim)
        lip_regions: (B, C, T, H, W)  (same as your VisualDataset output)
        """
        a = self.audio_encoder(mel)            # (B,512)
        c = self.cue_encoder(cue)              # (B,256)
        v = self.video_net.features(lip_regions)  # (B, video_feature_dim)

        fused = torch.cat([a, c, v], dim=1)
        x = self.fusion_proj(fused)
        logits = self.classifier(x)
        return logits
