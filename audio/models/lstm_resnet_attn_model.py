import torch
import torch.nn as nn
import torchvision.models as models


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        x: (batch, seq_len, hidden_dim)

        returns:
            weighted_sum: (batch, hidden_dim)
            weights:      (batch, seq_len)
        """
        weights = self.attn(x).squeeze(-1)          # (batch, seq_len)
        weights = torch.softmax(weights, dim=1)     # (batch, seq_len)

        # Weighted sum along sequence dimension
        weighted_sum = torch.sum(
            x * weights.unsqueeze(-1),              # (batch, seq_len, hidden_dim)
            dim=1                                   # → (batch, hidden_dim)
        )

        return weighted_sum, weights



class DeepAudioNetWithAttention(nn.Module):
    def __init__(self, num_classes=40, input_size=117):
        super(DeepAudioNetWithAttention, self).__init__()

        # ----------------------------------------------------------
        # 1) First BiLSTM across mel-band rows
        # Input:  each mel row is a sequence of 117 time frames
        # LSTM input:  (batch*80, 1, 117)
        # BiLSTM hidden size = 64 → output dim = 128
        # ----------------------------------------------------------
        self.initial_bilstm = nn.LSTM(
            input_size=input_size,   # 117
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # ----------------------------------------------------------
        # 2) ResNet18 for CNN feature extraction
        # Input to ResNet: (batch, 1, 80, 128)
        # Output:          (batch, 512)
        # ----------------------------------------------------------
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Identity()   # keep 512-dim pooled output

        # ----------------------------------------------------------
        # 3) FC projection before second BiLSTM
        # (batch, 512) → (batch, 256)
        # ----------------------------------------------------------
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # ----------------------------------------------------------
        # 4) Second BiLSTM (temporal modeling)
        # Input : (batch, seq_len=10, 256)
        # Output: (batch, 10, 256) since BiLSTM hidden=128 → 256-dim
        # ----------------------------------------------------------
        self.final_bilstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # ----------------------------------------------------------
        # 5) Attention layer across sequence (10 steps)
        # Input dim = 256 from the BiLSTM
        # Output    = (batch, 256)
        # ----------------------------------------------------------
        self.attention = Attention(input_dim=256)

        # ----------------------------------------------------------
        # 6) Final classifier
        # Input: 256
        # Output: num_classes
        # ----------------------------------------------------------
        self.classifier = nn.Linear(256, num_classes)


    def forward(self, x):
        """
        x: (batch, 80, 117)
            80  — mel-frequency bins
            117 — time frames
        """
        batch_size = x.size(0)

        # ----------------------------------------------------------
        # Step 1: First BiLSTM per mel-band
        # (batch, 80, 117) → (batch*80, 117) → add seq dim → (B*80, 1, 117)
        # Output: (batch*80, 1, 128)
        # ----------------------------------------------------------
        x1 = x.view(batch_size * 80, 117).unsqueeze(1)
        x1, _ = self.initial_bilstm(x1)

        # Reshape back to image-like form for ResNet
        # (batch*80, 1, 128) → (batch, 1, 80, 128)
        x1 = x1.squeeze(1).view(batch_size, 1, 80, -1)

        # ----------------------------------------------------------
        # Step 2: ResNet
        # Input: (batch, 1, 80, 128)
        # Output: (batch, 512)
        # ----------------------------------------------------------
        resnet_out = self.resnet(x1)

        # ----------------------------------------------------------
        # Step 3: FC projection
        # (batch, 512) → (batch, 256)
        # ----------------------------------------------------------
        fc_out = self.fc(resnet_out)

        # ----------------------------------------------------------
        # Step 4: Create sequence for second LSTM
        # Repeat 10 times → (batch, seq_len=10, 256)
        # ----------------------------------------------------------
        x_seq = fc_out.unsqueeze(1).repeat(1, 10, 1)

        # LSTM output: (batch, 10, 256)
        lstm_out, _ = self.final_bilstm(x_seq)

        # ----------------------------------------------------------
        # Step 5: Attention across the 10 steps
        # Output: (batch, 256)
        # ----------------------------------------------------------
        attn_out, _ = self.attention(lstm_out)

        # ----------------------------------------------------------
        # Step 6: Classification
        # Output: (batch, num_classes)
        # ----------------------------------------------------------
        return self.classifier(attn_out)
