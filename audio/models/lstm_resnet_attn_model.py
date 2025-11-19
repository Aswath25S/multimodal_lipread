import torch
import torch.nn as nn
import torchvision.models as models


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        weights = self.attn(x).squeeze(-1)      # (batch, seq_len)
        weights = torch.softmax(weights, dim=1) # (batch, seq_len)
        weighted_sum = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return weighted_sum, weights


class DeepAudioNetWithAttention(nn.Module):
    def __init__(self, num_classes=40, input_size=117, dropout_rate=0.3, use_batchnorm=True):
        super(DeepAudioNetWithAttention, self).__init__()

        self.use_bn = use_batchnorm

        # Step 1: First BiLSTM across mel-band rows
        self.initial_bilstm = nn.LSTM(
            input_size=input_size,   # 117
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Step 2: ResNet18 feature extractor
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()  # keep 512-dim output

        # Step 3: FC projection before second BiLSTM with BatchNorm + Dropout
        layers = [nn.Linear(512, 256)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(256))
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        self.fc = nn.Sequential(*layers)

        # Step 4: Second BiLSTM (temporal modeling)
        self.final_bilstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Step 5: Attention layer across sequence
        self.attention = Attention(input_dim=256)

        # Step 6: Final classifier
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Step 1: First BiLSTM per mel-band
        x1 = x.view(batch_size * 80, 117).unsqueeze(1)
        x1, _ = self.initial_bilstm(x1)

        # Reshape back to image-like form for ResNet
        x1 = x1.squeeze(1).view(batch_size, 1, 80, -1)

        # Step 2: ResNet
        resnet_out = self.resnet(x1)

        # Step 3: FC projection
        fc_out = self.fc(resnet_out)

        # Step 4: Create sequence for second LSTM
        x_seq = fc_out.unsqueeze(1).repeat(1, 10, 1)
        lstm_out, _ = self.final_bilstm(x_seq)

        # Step 5: Attention across the 10 steps
        attn_out, _ = self.attention(lstm_out)

        # Step 6: Classification
        return self.classifier(attn_out)
