import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torchvision.models as models


# --------------------------------------------------
# TimeDistributed (chunked) - processes frames in small chunks
# --------------------------------------------------
class TimeDistributedChunked(nn.Module):
    def __init__(self, module: nn.Module, chunk_size: int = 8):
        """
        module: maps (B*t, C, H, W) -> (B*t, feat)
        chunk_size: number of frames to process at once (per sample)
        """
        super().__init__()
        self.module = module
        self.chunk_size = int(chunk_size)

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.size()
        outputs = []
        device = x.device
        # iterate over temporal chunks (handles last smaller chunk)
        for i in range(0, T, self.chunk_size):
            t_end = min(i + self.chunk_size, T)
            frames = x[:, :, i:t_end]                 # (B, C, t_chunk, H, W)
            frames = frames.permute(0, 2, 1, 3, 4)   # (B, t_chunk, C, H, W)
            frames = frames.reshape(-1, C, H, W)     # (B * t_chunk, C, H, W)

            out = self.module(frames)                # (B * t_chunk, feat)
            # recover shape (B, t_chunk, feat)
            t_chunk = out.size(0) // B
            out = out.view(B, t_chunk, -1)
            outputs.append(out)

        return torch.cat(outputs, dim=1)  # (B, T, feat)


# --------------------------------------------------
# Safe wrapper to checkpoint the CNN forward
# --------------------------------------------------
class CNNWrapper(nn.Module):
    def __init__(self, cnn: nn.Module, use_checkpoint: bool = False):
        super().__init__()
        self.cnn = cnn
        self.use_checkpoint = bool(use_checkpoint)

    def forward(self, x):
        """
        Use checkpointing only when:
         - user requested it (use_checkpoint=True)
         - we are in training mode
         - input requires_grad==True (otherwise checkpoint warns and gradients are None)
        Also pass use_reentrant=False to satisfy future PyTorch requirements.
        """
        if self.use_checkpoint and self.training and getattr(x, "requires_grad", False):
            # pass use_reentrant explicitly to silence future warnings
            return checkpoint.checkpoint(self.cnn, x, use_reentrant=False)
        else:
            return self.cnn(x)


# --------------------------------------------------
# Video Encoder: MobileNet + BiLSTM (memory-optimized)
# --------------------------------------------------
class MobileNetLSTM(nn.Module):
    def __init__(self,
                 feature_dim: int = 256,
                 dropout: float = 0.3,
                 pretrained: bool = True,
                 freeze_backbone: bool = True,
                 use_checkpoint: bool = False,
                 td_chunk_size: int = 8,
                 lstm_num_layers: int = 1):
        """
        freeze_backbone: if True, MobileNet params won't require gradients (saves memory)
        use_checkpoint: if True, use torch.checkpoint for cnn (saves activations at cost of compute)
        td_chunk_size: frames per chunk processed by CNN
        lstm_num_layers: number of LSTM layers (use 1 to save memory)
        """
        super().__init__()

        # load MobileNet v2
        base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # remove classifier, keep features + pooling
        base.classifier = nn.Identity()
        cnn_seq = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # Optionally wrap with checkpointing
        self.cnn = CNNWrapper(cnn_seq, use_checkpoint=use_checkpoint)

        # optionally freeze backbone parameters (recommended for low VRAM)
        if freeze_backbone:
            for p in base.features.parameters():
                p.requires_grad = False

        # TimeDistributed that operates in small chunks
        self.td = TimeDistributedChunked(self.cnn, chunk_size=td_chunk_size)

        # LSTM - by default 1 layer to reduce memory usage; bidirectional to preserve feature_dim
        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=feature_dim // 2,
            num_layers=max(1, int(lstm_num_layers)),
            bidirectional=True,
            batch_first=True,
            dropout=dropout if int(lstm_num_layers) > 1 else 0.0
        )

        self.output_dim = feature_dim

    def forward(self, x):
        # x: (B, C, T, H, W)
        # After td: (B, T, 1280)
        x = self.td(x)
        # LSTM returns (B, T, feature_dim)
        x, _ = self.lstm(x)
        return x


# --------------------------------------------------
# Attention Module (unchanged)
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
        Q = self.query(q).unsqueeze(1)  # (B,1,D)
        K = self.key(k)                 # (B,T,D)
        V = self.value(k)               # (B,T,D)

        attn = torch.softmax((Q @ K.transpose(-1, -2)) * self.scale, dim=-1)  # (B,1,T)
        return (attn @ V).squeeze(1)  # (B,D)


# --------------------------------------------------
# Early Fusion Model (drop-in replacement)
# --------------------------------------------------
class EarlyAttentionFusion(nn.Module):
    def __init__(self,
                 num_classes,
                 cue_dim: int = 768,
                 pretrained: bool = True,
                 # memory-friendly options:
                 freeze_backbone: bool = True,
                 use_checkpoint: bool = True,
                 td_chunk_size: int = 8,
                 lstm_num_layers: int = 1):
        super().__init__()

        # video encoder (memory-optimized)
        self.video_encoder = MobileNetLSTM(
            feature_dim=256,
            dropout=0.3,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            use_checkpoint=use_checkpoint,
            td_chunk_size=td_chunk_size,
            lstm_num_layers=lstm_num_layers
        )

        # cue projection (unchanged)
        self.cue_proj = nn.Sequential(
            nn.Linear(cue_dim, 256),
            nn.ReLU()
        )

        self.attn = AttentionFusion(256)

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, cue, video):
        """
        cue: (B, cue_dim)
        video: (B, C, T, H, W)
        returns logits: (B, num_classes)
        """
        video_feats = self.video_encoder(video)   # (B, T, 256)
        cue_feat = self.cue_proj(cue)             # (B, 256)
        attended = self.attn(cue_feat, video_feats)  # (B, 256)
        return self.classifier(attended)
