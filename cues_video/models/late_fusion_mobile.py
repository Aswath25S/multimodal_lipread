import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torchvision.models as models


# --------------------------------------------------
# TimeDistributed (chunked + memory safe)
# --------------------------------------------------
class TimeDistributedChunked(nn.Module):
    def __init__(self, module, chunk_size=8):
        super().__init__()
        self.module = module
        self.chunk_size = int(chunk_size)

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.size()
        out_list = []

        for i in range(0, T, self.chunk_size):
            frames = x[:, :, i:i+self.chunk_size]          # (B,C,t,H,W)
            frames = frames.permute(0, 2, 1, 3, 4)         # (B,t,C,H,W)
            frames = frames.reshape(-1, C, H, W)           # (B*t,C,H,W)

            out = self.module(frames)                      # (B*t,1280)
            t = out.size(0) // B
            out = out.reshape(B, t, -1)                    # (B,t,1280)
            out_list.append(out)

        return torch.cat(out_list, dim=1)                  # (B,T,1280)


# --------------------------------------------------
# CNN wrapper (safe checkpointing)
# --------------------------------------------------
class CNNWrapper(nn.Module):
    def __init__(self, cnn, use_checkpoint=False):
        super().__init__()
        self.cnn = cnn
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint.checkpoint(self.cnn, x, use_reentrant=False)
        return self.cnn(x)


# --------------------------------------------------
# Video Encoder (memory optimized)
# --------------------------------------------------
class MobileNetLSTM(nn.Module):
    def __init__(self,
                 feature_dim=256,
                 dropout=0.3,
                 pretrained=True,
                 freeze_backbone=True,
                 use_checkpoint=False,
                 td_chunk_size=8,
                 lstm_layers=1):
        super().__init__()

        base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        base.classifier = nn.Identity()

        cnn = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.cnn = CNNWrapper(cnn, use_checkpoint=use_checkpoint)

        # ✅ Freeze CNN backbone (huge memory reduction)
        if freeze_backbone:
            for p in base.features.parameters():
                p.requires_grad = False

        # ✅ Chunked TimeDistributed
        self.td = TimeDistributedChunked(self.cnn, chunk_size=td_chunk_size)

        # ✅ Reduce LSTM from 2 layers → 1 layer (same output shape)
        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=feature_dim // 2,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        self.output_dim = feature_dim

    def forward(self, x):
        x = self.td(x)      # (B,T,1280)
        x, _ = self.lstm(x) # (B,T,256)
        return x


# --------------------------------------------------
# Late Fusion Model (DROP-IN REPLACEMENT)
# --------------------------------------------------
class LateAttentionFusion(nn.Module):
    def __init__(self,
                 num_classes,
                 cue_dim=768,
                 pretrained=True,
                 freeze_backbone=True,
                 use_checkpoint=True,
                 td_chunk_size=8,
                 lstm_layers=1):
        super().__init__()

        self.video_encoder = MobileNetLSTM(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            use_checkpoint=use_checkpoint,
            td_chunk_size=td_chunk_size,
            lstm_layers=lstm_layers
        )

        self.video_head = nn.Linear(256, num_classes)

        self.cue_head = nn.Sequential(
            nn.Linear(cue_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self.attn = nn.Sequential(
            nn.Linear(num_classes * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, cue, video):
        # video feature from last timestep
        v_feat = self.video_encoder(video)[:, -1, :]
        v_logits = self.video_head(v_feat)
        c_logits = self.cue_head(cue)

        weights = self.attn(torch.cat([v_logits, c_logits], dim=1))
        wv, wc = weights[:, 0:1], weights[:, 1:2]

        return wv * v_logits + wc * c_logits
