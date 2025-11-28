# models/fusion_late_resnet.py
import torch
import torch.nn as nn
import torchvision.models as models

class AudioModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        resnet.fc = nn.Identity()
        self.encoder = resnet
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        return self.classifier(x)


class CueModel(nn.Module):
    def __init__(self, input_dim=768, num_classes=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class LateFusionAttentionResNet(nn.Module):
    def __init__(self, num_classes, cue_dim=768):
        super().__init__()
        self.audio_model = AudioModel(num_classes)
        self.cue_model = CueModel(input_dim=cue_dim, num_classes=num_classes)

        # learnable attention weights over modalities
        self.attn_weights = nn.Parameter(torch.ones(2))

    def forward(self, mel, cue):
        audio_logits = self.audio_model(mel)
        cue_logits = self.cue_model(cue)

        weights = torch.softmax(self.attn_weights, dim=0)
        fused_logits = weights[0]*audio_logits + weights[1]*cue_logits
        return fused_logits
