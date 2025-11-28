# models/fusion_early.py
import torch
import torch.nn as nn
import torchvision.models as models

class AudioEncoder(nn.Module):
    def __init__(self, width_mult=1.0):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        mobilenet.features[0][0] = nn.Conv2d(1, int(32*width_mult), kernel_size=3, stride=2, padding=1, bias=False)
        self.encoder = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.output_dim = int(1280 * width_mult)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class CueEncoder(nn.Module):
    def __init__(self, input_dim=768, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output_dim = output_dim

    def forward(self, x):
        return self.net(x)


class EarlyFusionAttentionMobile(nn.Module):
    def __init__(self, num_classes, cue_dim=768, audio_width_mult=1.0):
        super().__init__()
        self.audio_encoder = AudioEncoder(width_mult=audio_width_mult)
        self.cue_encoder = CueEncoder(input_dim=cue_dim, output_dim=128)

        fusion_dim = self.audio_encoder.output_dim + self.cue_encoder.output_dim

        # Attention layer
        self.attn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, mel, cue):
        audio_feat = self.audio_encoder(mel)
        cue_feat = self.cue_encoder(cue)
        fused = torch.cat([audio_feat, cue_feat], dim=1)  # (B, fusion_dim)

        # Attention weighting
        attn_weights = torch.softmax(self.attn(fused), dim=0)
        fused = fused * attn_weights

        out = self.classifier(fused)
        return out
