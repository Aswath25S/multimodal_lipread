import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class LSTMResNetWithTransformer(nn.Module):
    def __init__(self, num_classes=40, input_size=117, transformer_dim=256,
                 num_heads=4, num_layers=2, seq_len=10, dropout_rate=0.3, use_batchnorm=True):
        super(LSTMResNetWithTransformer, self).__init__()

        self.seq_len = seq_len
        self.use_bn = use_batchnorm

        # BiLSTM over mel-frequency rows
        self.initial_bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # ResNet18 feature extractor
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()  # output 512-dim pooled vector

        # FC projection to Transformer dimension with BatchNorm + Dropout
        layers = [nn.Linear(512, transformer_dim)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(transformer_dim))
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        self.fc = nn.Sequential(*layers)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(dim=transformer_dim, max_len=seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classifier
        self.classifier = nn.Linear(transformer_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Ensure input shape: (batch, 80, 117)
        if x.dim() == 4:  # (batch, 1, 80, 117)
            x = x.squeeze(1)

        mel_bins = x.shape[1]
        time_steps = x.shape[2]

        # === BiLSTM over mel rows ===
        x1 = x.view(batch_size * mel_bins, time_steps).unsqueeze(1)
        x1, _ = self.initial_bilstm(x1)

        # Restore 2D layout: (batch, 1, 80, 128)
        x1 = x1.squeeze(1).view(batch_size, 1, mel_bins, -1)

        # === ResNet feature extraction ===
        resnet_out = self.resnet(x1)

        # === Project to Transformer dimension ===
        fc_out = self.fc(resnet_out)

        # === Prepare sequence for Transformer ===
        x_seq = fc_out.unsqueeze(1).repeat(1, self.seq_len, 1)

        # Add positional encoding
        x_seq = self.pos_encoder(x_seq)

        # === Transformer encoder ===
        x_encoded = self.transformer(x_seq)

        # Global average pooling
        pooled = x_encoded.mean(dim=1)

        # Classification
        return self.classifier(pooled)
