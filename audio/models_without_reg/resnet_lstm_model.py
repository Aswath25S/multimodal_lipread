import torch
import torch.nn as nn
import torchvision.models as models

class AudioResNetLSTM(nn.Module):
    def __init__(self, num_classes=40, lstm_hidden=128, lstm_layers=2):
        super(AudioResNetLSTM, self).__init__()

        # Load pretrained 2D ResNet and adapt for single-channel input
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify first conv layer to accept 1-channel input instead of 3
        # Input expected: (batch, 1, 80, 117)
        self.resnet.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=7, stride=2, padding=3,
            bias=False
        )

        # Remove last FC layer → ResNet now outputs a (batch, 512) embedding
        self.resnet.fc = nn.Identity()

        # ------------------------------------------------------------
        # 🔹 LSTM block after ResNet
        # Input to LSTM: (batch, seq_len=1, 512)
        # Output: (batch, 1, 2*lstm_hidden)
        # ------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True
        )

        # Final classifier after LSTM output
        # LSTM output dimension = 2 * lstm_hidden
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        x: input tensor of shape (batch_size, 80, 117)
        """
        # Add channel dimension → (batch, 1, 80, 117)
        x = x.unsqueeze(1)

        # ResNet backbone → (batch, 512)
        features = self.resnet(x)

        # Prepare for LSTM: add sequence dimension → (batch, 1, 512)
        lstm_input = features.unsqueeze(1)

        # LSTM output → (batch, 1, 2*lstm_hidden)
        lstm_out, _ = self.lstm(lstm_input)

        # Take last time step → (batch, 2*lstm_hidden)
        final_features = lstm_out[:, -1, :]

        # Classify → (batch, num_classes)
        output = self.classifier(final_features)

        return output
