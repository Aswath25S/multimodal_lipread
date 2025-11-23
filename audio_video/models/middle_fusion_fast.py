import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MidFusionFast(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        self.audio_cnn = nn.Sequential(
            nn.Conv2d(config.get("dataset.audio_channels",1),16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.audio_fc = nn.Linear(16*40*58, config.get("model.audio_feature_dim",128))

        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        base.classifier = nn.Identity()
        self.video_cnn = base
        self.video_lstm = nn.LSTM(576,128,1,batch_first=True,bidirectional=True)

        fusion_dim = 128 + 256  # video LSTM 128*2 + audio 128
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, audio, video):
        a = audio.unsqueeze(1)
        a = self.audio_cnn(a).view(a.size(0), -1)
        a = self.audio_fc(a)

        B,C,T,H,W = video.shape
        v = video.permute(0,2,1,3,4).contiguous().view(B*T,C,H,W)
        v = self.video_cnn(v).view(B,T,-1)
        v,_ = self.video_lstm(v)
        v = v[:,-1]

        fused = torch.cat([a,v], dim=1)
        return self.classifier(fused)

def create_mid_fusion_fast(num_classes, config):
    return MidFusionFast(num_classes, config)
