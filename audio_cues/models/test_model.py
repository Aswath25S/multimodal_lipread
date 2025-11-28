import torch
import torch.nn as nn
import torchvision.models as models


# ================================
# AUDIO ENCODER (ResNet on Mel)
# ================================
class AudioEncoder(nn.Module):
    def __init__(self, input_size=117):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Adapt input to 1-channel spectrogram
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove final classifier
        resnet.fc = nn.Identity()

        self.encoder = resnet
        self.output_dim = 512

    def forward(self, x):
        # x: (B, 80, 117)
        x = x.unsqueeze(1)          # (B,1,80,117)
        return self.encoder(x)      # (B,512)


# ================================
# CUE ENCODER (Dense Projection)
# ================================
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
        return self.net(x)   # (B,256)


# ================================
# MULTIMODAL FUSION MODEL
# ================================
class MultimodalNet(nn.Module):
    def __init__(self, num_classes, cue_dim=768, audio_dim=512):
        super().__init__()

        self.audio_encoder = AudioEncoder()
        self.cue_encoder = CueEncoder(input_dim=cue_dim)

        fusion_dim = audio_dim + 256

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, mel, cue):
        # mel: (B,80,117)
        # cue: (B,768)

        audio_feat = self.audio_encoder(mel)
        cue_feat   = self.cue_encoder(cue)

        fused = torch.cat([audio_feat, cue_feat], dim=1)

        return self.classifier(fused)
