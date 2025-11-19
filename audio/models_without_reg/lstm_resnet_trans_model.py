import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix of shape (max_len, dim)
        pe = torch.zeros(max_len, dim)

        # Position indices: (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Compute sin/cos frequency scaling
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))

        # Apply sin to even dimensions, cos to odd dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension → (1, max_len, dim)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        x: (batch, seq_len, dim)
        Output: x + positional encoding → (batch, seq_len, dim)
        """
        x = x + self.pe[:, :x.size(1)]
        return x


class LSTMResNetWithTransformer(nn.Module):
    def __init__(self, num_classes=40, input_size=117,
                 transformer_dim=256, num_heads=4, num_layers=2, seq_len=10):
        super(LSTMResNetWithTransformer, self).__init__()

        self.seq_len = seq_len

        # BiLSTM applied across mel-frequency rows
        # Input: each row = (117 timesteps)
        # We reshape input so LSTM sees (batch*80, 1, 117)
        # Output per row → (batch*80, 1, 128)  [because hidden=64, bidirectional=128]
        self.initial_bilstm = nn.LSTM(
            input_size=117,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # 2D ResNet backbone
        # Input expected: (batch, 1, 80, 128)
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()  # Output → pooled 512-dim feature

        # Map ResNet output (512) into Transformer model dimension
        self.fc = nn.Sequential(
            nn.Linear(512, transformer_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Add positional information for Transformer sequence
        self.pos_encoder = PositionalEncoding(
            dim=transformer_dim,
            max_len=seq_len
        )

        # Transformer encoder for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classification output: (batch, num_classes)
        self.classifier = nn.Linear(transformer_dim, num_classes)


    def forward(self, x):
        """
        Expected input:
            x: (batch, 80, 117)
        where:
            80  → mel-frequency rows
            117 → time frames
        """

        batch_size = x.size(0)

        # If input accidentally includes a channel dim, squeeze it
        # Ensure shape: (batch, 80, 117)
        if x.dim() == 4:  # (batch, 1, 80, 117)
            x = x.squeeze(1)

        mel_bins = x.shape[1]   # 80
        time_steps = x.shape[2] # 117

        # === BiLSTM over frequency rows ===
        # Reshape to process each mel-band independently:
        # x → (batch*80, 117)
        x1 = x.view(batch_size * mel_bins, time_steps).unsqueeze(1)  # (batch*80, 1, 117)

        # LSTM output: (batch*80, 1, 128)
        x1, _ = self.initial_bilstm(x1)

        # Remove seq dim and restore spatial structure
        # New shape: (batch, 1, 80, 128)
        x1 = x1.squeeze(1).view(batch_size, 1, 80, -1)

        # === ResNet feature extraction ===
        # Input: (batch, 1, 80, 128)
        # Output: (batch, 512)
        resnet_out = self.resnet(x1)

        # === Project to Transformer dimension ===
        # (batch, 512) → (batch, transformer_dim)
        fc_out = self.fc(resnet_out)

        # === Prepare repeated sequence for Transformer ===
        # Expand to length `seq_len`
        # (batch, transformer_dim) → (batch, seq_len, transformer_dim)
        x_seq = fc_out.unsqueeze(1).repeat(1, self.seq_len, 1)

        # Add positional encoding
        x_seq = self.pos_encoder(x_seq)  # (batch, seq_len, transformer_dim)

        # === Transformer encoder ===
        # Output: (batch, seq_len, transformer_dim)
        x_encoded = self.transformer(x_seq)

        # Global average pooling across sequence length
        # (batch, seq_len, transformer_dim) → (batch, transformer_dim)
        pooled = x_encoded.mean(dim=1)

        # Final classifier
        # Output: (batch, num_classes)
        return self.classifier(pooled)
