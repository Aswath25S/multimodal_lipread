# models/fusion_late.py
import torch
import torch.nn as nn
import torchvision.models as models

class AudioEncoder(nn.Module):
    def __init__(self, width_mult=1.0):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        mobilenet.features[0][0] = nn.Conv2d(1, int(32*width_mult), 3, 2, 1, bias=False)
        self.encoder = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.output_dim = int(1280*width_mult)
        self.classifier = nn.Sequential(
            nn.Linear(self.output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 100)  # placeholder, will override
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class CueEncoder(nn.Module):
    def __init__(self, input_dim=768, output_dim=128, num_classes=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class LateFusionAttentionMobile(nn.Module):
    def __init__(self, num_classes, cue_dim=768, audio_width_mult=1.0):
        super().__init__()
        self.audio_model = AudioEncoder(width_mult=audio_width_mult)
        self.cue_model = CueEncoder(input_dim=cue_dim, output_dim=128, num_classes=num_classes)

        # Attention weights over modalities
        self.attn_weights = nn.Parameter(torch.ones(2))  # learnable [w_audio, w_cue]

    def forward(self, mel, cue):
        audio_logits = self.audio_model(mel)
        cue_logits = self.cue_model(cue)

        # Softmax over attention weights
        weights = torch.softmax(self.attn_weights, dim=0)
        fused_logits = weights[0]*audio_logits + weights[1]*cue_logits
        return fused_logits
