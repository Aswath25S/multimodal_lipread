# models/fusion_middle_resnet.py
import torch
import torch.nn as nn
import torchvision.models as models

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        resnet.fc = nn.Identity()
        self.encoder = resnet
        self.output_dim = 512

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.encoder(x)


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


class MiddleFusionAttentionResNet(nn.Module):
    def __init__(self, num_classes, cue_dim=768):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.cue_encoder = CueEncoder(input_dim=cue_dim, output_dim=128)

        fusion_dim = self.audio_encoder.output_dim + self.cue_encoder.output_dim

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=4, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, mel, cue):
        audio_feat = self.audio_encoder(mel)
        cue_feat = self.cue_encoder(cue)
        fused = torch.cat([audio_feat, cue_feat], dim=1).unsqueeze(1)  # (B,1,fusion_dim)

        attn_out, _ = self.cross_attn(fused, fused, fused)
        attn_out = attn_out.squeeze(1)
        return self.classifier(attn_out)
