import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torchvision.models as models


# --------------------------------------------------
# Attention block (unchanged)
# --------------------------------------------------
class AttentionFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, feats):
        stacked = torch.stack(feats, dim=1)
        scores = self.attn(stacked).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return (stacked * weights.unsqueeze(-1)).sum(dim=1), weights


# --------------------------------------------------
# Memory-safe TimeDistributed (chunked)
# --------------------------------------------------
class TimeDistributedChunked(nn.Module):
    def __init__(self, module, chunk_size=4):
        super().__init__()
        self.module = module
        self.chunk_size = int(chunk_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        out_chunks = []

        for i in range(0, T, self.chunk_size):
            t_end = min(i + self.chunk_size, T)
            frames = x[:, :, i:t_end]
            frames = frames.permute(0,2,1,3,4).reshape(-1, C, H, W)

            y = self.module(frames)
            t = y.size(0) // B
            out_chunks.append(y.view(B, t, -1))

        return torch.cat(out_chunks, dim=1)


# --------------------------------------------------
# Safe checkpoint wrapper
# --------------------------------------------------
class SafeCheckpoint(nn.Module):
    def __init__(self, module, enabled=True):
        super().__init__()
        self.module = module
        self.enabled = enabled

    def forward(self, x):
        if self.enabled and self.training and x.requires_grad:
            return checkpoint.checkpoint(self.module, x, use_reentrant=False)
        return self.module(x)


# --------------------------------------------------
# Video Encoder (ResNet + small LSTM)
# --------------------------------------------------
class ResNetLSTM(nn.Module):
    def __init__(self, feature_dim=256, pretrained=True):
        super().__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        resnet.fc = nn.Identity()

        # Freeze backbone
        for p in resnet.parameters():
            p.requires_grad = False

        # Freeze BatchNorm layers
        def freeze_bn(m):
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        resnet.apply(freeze_bn)

        # CNN with checkpointing
        self.cnn = SafeCheckpoint(nn.Sequential(resnet), enabled=True)

        # Chunked TimeDistributed
        self.td = TimeDistributedChunked(self.cnn, chunk_size=4)

        # Reduced LSTM depth (keeps same output size)
        self.lstm = nn.LSTM(
            512,
            feature_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.0
        )

        self.output_dim = feature_dim

    def forward(self, x):
        x = self.td(x)
        x, _ = self.lstm(x)
        return x[:, -1, :]


# --------------------------------------------------
# Audio Encoder (frozen ResNet)
# --------------------------------------------------
class AudioEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        net = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        net.fc = nn.Identity()

        # Freeze params
        for p in net.parameters():
            p.requires_grad = False

        # BatchNorm to eval
        def freeze_bn(m):
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        net.apply(freeze_bn)

        self.enc = SafeCheckpoint(net, enabled=True)
        self.output_dim = 512

    def forward(self, x):
        return self.enc(x.unsqueeze(1))


# --------------------------------------------------
# Cue Encoder (unchanged)
# --------------------------------------------------
class CueEncoder(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.output_dim = 256

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------
# MIDDLE FUSION MODEL (DROP-IN)
# --------------------------------------------------
class MultimodalAttentionMiddleResNet(nn.Module):
    def __init__(self, num_classes, cue_dim=768, video_cfg=None, pretrained=True):
        super().__init__()

        self.audio = AudioEncoder(pretrained)
        self.cue = CueEncoder(cue_dim)

        vdim = int(video_cfg.get("model", {}).get("feature_dim", 256)) if video_cfg else 256
        self.video = ResNetLSTM(vdim, pretrained)

        self.ap = nn.Linear(512, 256)
        self.vp = nn.Linear(vdim, 256)

        self.attn = AttentionFusion(256)

        self.cls = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, mel, cue, lip):
        a = self.ap(self.audio(mel))
        c = self.cue(cue)
        v = self.vp(self.video(lip))

        fused, _ = self.attn([a, c, v])
        return self.cls(fused)
