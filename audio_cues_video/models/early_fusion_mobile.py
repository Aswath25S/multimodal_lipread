import torch
import torch.nn as nn
import torchvision.models as models


# --------------------------------------------------
# Attention block
# --------------------------------------------------
class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_modalities=3):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, feats):
        # feats: list of (B,D)
        stacked = torch.stack(feats, dim=1)     # (B,M,D)
        scores = self.attn(stacked).squeeze(-1) # (B,M)
        weights = torch.softmax(scores, dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights


# --------------------------------------------------
# TimeDistributed
# --------------------------------------------------
class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        B,C,T,H,W = x.size()
        x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        out = self.module(x)
        out = out.reshape(B,T,-1)
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
            nn.AdaptiveAvgPool2d((1,1)),
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
        x,_ = self.lstm(x)
        return x[:, -1, :]


# --------------------------------------------------
# Audio Encoder
# --------------------------------------------------
class AudioEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
        net.fc = nn.Identity()
        self.encoder = net
        self.output_dim = 512

    def forward(self, x):
        return self.encoder(x.unsqueeze(1))


# --------------------------------------------------
# Cue Encoder
# --------------------------------------------------
class CueEncoder(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,256),
            nn.ReLU()
        )
        self.output_dim = 256

    def forward(self,x):
        return self.net(x)


# --------------------------------------------------
# EARLY ATTENTION MODEL
# --------------------------------------------------
class MultimodalAttentionEarly(nn.Module):
    def __init__(self, num_classes, cue_dim=768, video_cfg=None, pretrained=True):
        super().__init__()

        self.audio = AudioEncoder(pretrained)
        self.cue   = CueEncoder(cue_dim)

        vdim = 256
        if video_cfg:
            vdim = int(video_cfg.get("model", {}).get("feature_dim", vdim))

        self.video = MobileNetLSTM(feature_dim=vdim, pretrained=pretrained)

        self.ap = nn.Linear(512,256)
        self.vp = nn.Linear(vdim,256)
        self.cp = nn.Linear(256,256)

        self.attn = AttentionFusion(256)

        self.classifier = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, mel, cue, lip):
        a = self.ap(self.audio(mel))
        c = self.cp(self.cue(cue))
        v = self.vp(self.video(lip))

        fused,_ = self.attn([a,c,v])
        return self.classifier(fused)
