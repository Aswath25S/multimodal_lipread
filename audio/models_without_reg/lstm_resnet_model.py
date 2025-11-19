import torch
import torch.nn as nn
import torchvision.models as models

class LSTMResNet(nn.Module):
    def __init__(self, num_classes=40, input_size=117):
        super(LSTMResNet, self).__init__()

        # --------------------------------------------------------------
        # 1) First BiLSTM across mel-frequency rows (each row = a sequence)
        # Input to LSTM:  (batch*80, 1, 117)
        # Hidden size = 64 → bidirectional → output dim = 128 per mel row
        # --------------------------------------------------------------
        self.initial_bilstm = nn.LSTM(
            input_size=input_size,  # 117 time steps per mel band
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # --------------------------------------------------------------
        # 2) ResNet18 for spatial feature extraction
        # Input shape to CNN: (batch, 1, 80, 128)
        # Output (after pooling): (batch, 512)
        # --------------------------------------------------------------
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Replace RGB conv with 1-channel conv for spectrogram input
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove classification head → keep 512-dim pooled features
        self.resnet.fc = nn.Identity()

        # --------------------------------------------------------------
        # 3) Fully connected transformation before second BiLSTM
        # Input:  (batch, 512)
        # Output: (batch, 256)
        # --------------------------------------------------------------
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # --------------------------------------------------------------
        # 4) Second BiLSTM over the ResNet output features
        # Input sequence shape: (batch, seq_len=1, 256)
        # Bidirectional hidden size = 128 → output dim = 256
        # --------------------------------------------------------------
        self.final_bilstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # --------------------------------------------------------------
        # 5) Final classifier → (batch, num_classes)
        # Input: 256-dim from last LSTM time step
        # --------------------------------------------------------------
        self.classifier = nn.Linear(2 * 128, num_classes)


    def forward(self, x):
        """
        Expected input:
            x: (batch_size, 80, 117)
        Where:
            80  — mel-frequency bins
            117 — time frames
        """
        batch_size = x.size(0)

        # --------------------------------------------------------------
        # Step 1: First BiLSTM along mel rows
        # Reshape so each mel row is treated as an LSTM sequence:
        # x → (batch*80, 117)
        # Add sequence dimension → (batch*80, 1, 117)
        # Output → (batch*80, 1, 128)
        # --------------------------------------------------------------
        x1 = x.view(batch_size * 80, 117).unsqueeze(1)
        x1, _ = self.initial_bilstm(x1)

        # Remove LSTM's sequence dimension and restore 2D layout:
        # (batch*80, 1, 128) → (batch, 1, 80, 128)
        x1 = x1.squeeze(1).view(batch_size, 1, 80, -1)

        # --------------------------------------------------------------
        # Step 2: ResNet feature extraction
        # Input to ResNet: (batch, 1, 80, 128)
        # Output pooled vector: (batch, 512)
        # --------------------------------------------------------------
        resnet_out = self.resnet(x1)

        # --------------------------------------------------------------
        # Step 3: Dense projection to 256-dim
        # (batch, 512) → (batch, 256)
        # --------------------------------------------------------------
        fc_out = self.fc(resnet_out)

        # --------------------------------------------------------------
        # Step 4: Expand features into a "sequence" for second LSTM
        # (batch, 256) → (batch, 1, 256)
        # --------------------------------------------------------------
        seq_input = fc_out.unsqueeze(1)

        # LSTM output: (batch, 1, 256)
        lstm_out, _ = self.final_bilstm(seq_input)

        # Take final time step → (batch, 256)
        final_out = lstm_out[:, -1, :]

        # --------------------------------------------------------------
        # Step 5: Final classifier → (batch, num_classes)
        # --------------------------------------------------------------
        return self.classifier(final_out)
