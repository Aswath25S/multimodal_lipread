import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torchvision.models as models


# --------------------------------------------------
# Attention block (unchanged API)
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
# Chunked TimeDistributed (memory-safe)
# --------------------------------------------------
class TimeDistributedChunked(nn.Module):
    def __init__(self, module: nn.Module, chunk_size: int = 4):
        """
        module: maps (B*t, C, H, W) -> (B*t, feat)
        chunk_size: how many frames per chunk (per sample) to process at once
        """
        super().__init__()
        self.module = module
        self.chunk_size = int(chunk_size)

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        out_chunks = []
        for i in range(0, T, self.chunk_size):
            t_end = min(i + self.chunk_size, T)
            frames = x[:, :, i:t_end]                 # (B, C, t_chunk, H, W)
            frames = frames.permute(0, 2, 1, 3, 4)   # (B, t_chunk, C, H, W)
            frames = frames.reshape(-1, C, H, W)     # (B * t_chunk, C, H, W)

            out = self.module(frames)                # (B * t_chunk, feat)
            t_chunk = out.size(0) // B
            out = out.view(B, t_chunk, -1)           # (B, t_chunk, feat)
            out_chunks.append(out)

        return torch.cat(out_chunks, dim=1)          # (B, T, feat)


# --------------------------------------------------
# Safe checkpoint wrapper
# --------------------------------------------------
class SafeCheckpoint(nn.Module):
    def __init__(self, module: nn.Module, enabled: bool = True):
        super().__init__()
        self.module = module
        self.enabled = bool(enabled)

    def forward(self, x):
        # Only checkpoint when it's useful: enabled, training, and input requires grad.
        if self.enabled and self.training and getattr(x, "requires_grad", False):
            return checkpoint.checkpoint(self.module, x, use_reentrant=False)
        return self.module(x)


# --------------------------------------------------
# Video Encoder: MobileNet + (reduced) BiLSTM
# --------------------------------------------------
class MobileNetLSTM(nn.Module):
    def __init__(self, feature_dim=256, dropout=0.3, pretrained=True,
                 freeze_backbone: bool = True, td_chunk_size: int = 4, use_checkpoint: bool = True):
        super().__init__()

        base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        base.classifier = nn.Identity()

        # CNN block that returns (N, 1280)
        cnn_seq = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # Wrap CNN in SafeCheckpoint (only recomputes when grads needed)
        self.cnn = SafeCheckpoint(cnn_seq, enabled=use_checkpoint)

        # Optionally freeze backbone to save memory
        if freeze_backbone:
            for p in base.features.parameters():
                p.requires_grad = False

        # Chunked TimeDistributed to avoid sending B*T images at once
        self.td = TimeDistributedChunked(self.cnn, chunk_size=td_chunk_size)

        # Use 1 LSTM layer (bidirectional) to keep same output size but reduce memory
        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=feature_dim // 2,
            num_layers=1,  # reduced from 2 -> major memory win
            bidirectional=True,
            batch_first=True,
            dropout=dropout if 1 > 1 else 0.0
        )

        self.output_dim = feature_dim

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.td(x)              # (B, T, 1280)
        x, _ = self.lstm(x)         # (B, T, feature_dim)
        return x[:, -1, :]          # (B, feature_dim)


# --------------------------------------------------
# Audio Encoder (ResNet) — frozen + BN eval + checkpoint wrapper
# --------------------------------------------------
class AudioEncoder(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone: bool = True, use_checkpoint: bool = True):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        net.fc = nn.Identity()

        # Freeze parameters to reduce memory
        if freeze_backbone:
            for p in net.parameters():
                p.requires_grad = False

            # Put BatchNorm layers into eval mode to avoid running stats updates / extra memory
            def _freeze_bn(m):
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            net.apply(_freeze_bn)

        # Wrap in SafeCheckpoint
        self.encoder = SafeCheckpoint(net, enabled=use_checkpoint)
        self.output_dim = 512

    def forward(self, x):
        # expect mel input shaped (B, freq, time) or similar; original code did x.unsqueeze(1)
        return self.encoder(x.unsqueeze(1))


# --------------------------------------------------
# Cue Encoder (kept same, but BN could be problematic for tiny batches)
# --------------------------------------------------
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


# --------------------------------------------------
# EARLY ATTENTION MODEL (drop-in)
# --------------------------------------------------
class MultimodalAttentionEarly(nn.Module):
    def __init__(self, num_classes, cue_dim=768, video_cfg=None, pretrained=True):
        super().__init__()

        # audio, cue, video encoders
        self.audio = AudioEncoder(pretrained=pretrained, freeze_backbone=True, use_checkpoint=True)
        self.cue = CueEncoder(cue_dim)

        vdim = 256
        if video_cfg:
            vdim = int(video_cfg.get("model", {}).get("feature_dim", vdim))

        self.video = MobileNetLSTM(feature_dim=vdim, pretrained=pretrained,
                                  freeze_backbone=True, td_chunk_size=4, use_checkpoint=True)

        # projection heads (same dims as original)
        self.ap = nn.Linear(512, 256)
        self.vp = nn.Linear(vdim, 256)
        self.cp = nn.Linear(256, 256)

        self.attn = AttentionFusion(256)

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, mel, cue, lip):
        a = self.ap(self.audio(mel))     # (B,256)
        c = self.cp(self.cue(cue))       # (B,256)
        v = self.vp(self.video(lip))     # (B,256)

        fused, _ = self.attn([a, c, v])  # (B,256)
        return self.classifier(fused)    # (B,num_classes)
