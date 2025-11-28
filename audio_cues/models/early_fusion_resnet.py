# models/fusion_early_resnet.py
import torch
import torch.nn as nn
import torchvision.models as models

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)  # adapt to 1-channel
        resnet.fc = nn.Identity()
        self.encoder = resnet
        self.output_dim = 512

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.encoder(x)  # (B,512)


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


class EarlyFusionAttentionResNet(nn.Module):
    def __init__(self, num_classes, cue_dim=768):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.cue_encoder = CueEncoder(input_dim=cue_dim, output_dim=128)

        fusion_dim = self.audio_encoder.output_dim + self.cue_encoder.output_dim

        # Attention
        self.attn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, mel, cue):
        audio_feat = self.audio_encoder(mel)
        cue_feat = self.cue_encoder(cue)
        fused = torch.cat([audio_feat, cue_feat], dim=1)

        attn_weights = torch.softmax(self.attn(fused), dim=0)
        fused = fused * attn_weights

        return self.classifier(fused)
