import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torchvision.models as models


# --------------------------------------------------
# TimeDistributed (chunked) - memory friendly
# --------------------------------------------------
class TimeDistributedChunked(nn.Module):
    def __init__(self, module: nn.Module, chunk_size: int = 8):
        """
        module: maps (B*t, C, H, W) -> (B*t, feat)
        chunk_size: how many frames per chunk to process at once
        """
        super().__init__()
        self.module = module
        self.chunk_size = int(chunk_size)

    def forward(self, x):
        # Expect x: (B, C, T, H, W)
        B, C, T, H, W = x.size()
        outputs = []
        for i in range(0, T, self.chunk_size):
            t_end = min(i + self.chunk_size, T)
            frames = x[:, :, i:t_end]                # (B, C, t_chunk, H, W)
            frames = frames.permute(0, 2, 1, 3, 4)  # (B, t_chunk, C, H, W)
            frames = frames.reshape(-1, C, H, W)    # (B*t_chunk, C, H, W)

            out = self.module(frames)               # (B*t_chunk, feat)
            t_chunk = out.size(0) // B
            out = out.view(B, t_chunk, -1)          # (B, t_chunk, feat)
            outputs.append(out)

        return torch.cat(outputs, dim=1)            # (B, T, feat)


# --------------------------------------------------
# Safe CNN wrapper (checkpointing only when useful)
# --------------------------------------------------
class CNNWrapper(nn.Module):
    def __init__(self, cnn: nn.Module, use_checkpoint: bool = False):
        super().__init__()
        self.cnn = cnn
        self.use_checkpoint = bool(use_checkpoint)

    def forward(self, x):
        # Only checkpoint when requested, training, and input requires grad.
        if self.use_checkpoint and self.training and getattr(x, "requires_grad", False):
            return checkpoint.checkpoint(self.cnn, x, use_reentrant=False)
        return self.cnn(x)


# --------------------------------------------------
# Video Encoder: MobileNet + LSTM (memory-optimized)
# --------------------------------------------------
class MobileNetLSTM(nn.Module):
    def __init__(self, feature_dim=256, dropout=0.3, pretrained=True):
        """
        Memory-optimized MobileNet + LSTM encoder.
        Default choices favor low VRAM:
         - freeze MobileNet features
         - chunk_size = 8
         - lstm layers = 1
         - use checkpointing by default
        """
        super().__init__()

        base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        base.classifier = nn.Identity()

        # CNN sequence: features -> pooled -> flattened
        cnn_seq = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # wrapper with safe checkpointing (only active when grad needed)
        self.cnn = CNNWrapper(cnn_seq, use_checkpoint=True)

        # Freeze backbone parameters to avoid storing gradients/activations for the CNN
        for p in base.features.parameters():
            p.requires_grad = False

        # chunked TimeDistributed to avoid passing B*T frames at once
        self.td = TimeDistributedChunked(self.cnn, chunk_size=8)

        # Use 1 LSTM layer (bidirectional) to preserve feature_dim while reducing memory
        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=feature_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.0
        )

        self.output_dim = feature_dim

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.td(x)        # (B, T, 1280)
        x, _ = self.lstm(x)   # (B, T, feature_dim)
        return x


# --------------------------------------------------
# Attention (unchanged)
# --------------------------------------------------
class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, q, k):
        # q: (B,D), k: (B,T,D)
        Q = self.query(q).unsqueeze(1)
        K = self.key(k)
        V = self.value(k)
        attn = torch.softmax((Q @ K.transpose(-1, -2)) * self.scale, dim=-1)
        return (attn @ V).squeeze(1)


# --------------------------------------------------
# Middle Fusion Model (drop-in replacement)
# --------------------------------------------------
class MiddleAttentionFusion(nn.Module):
    def __init__(self, num_classes, cue_dim=768, pretrained=True):
        """
        Signature kept identical to your original.
        Internally the video encoder is memory-optimized.
        """
        super().__init__()

        self.video_encoder = MobileNetLSTM(pretrained=pretrained)

        self.cue_encoder = nn.Sequential(
            nn.Linear(cue_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.attn = AttentionFusion(256)

        self.fusion = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, cue, video):
        """
        cue: (B, cue_dim)
        video: (B, C, T, H, W)
        """
        video_feats = self.video_encoder(video)   # (B, T, 256)
        video_last = video_feats[:, -1, :]        # (B, 256)

        cue_feat = self.cue_encoder(cue)          # (B, 256)

        attended = self.attn(cue_feat, video_feats)  # (B, 256)
        fused = torch.cat([video_last, attended], dim=1)  # (B, 512)

        return self.classifier(self.fusion(fused))  # (B, num_classes)
