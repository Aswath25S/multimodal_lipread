import torch
import torch.nn as nn
import torchvision.models as models

class AudioResNetLSTM(nn.Module):
    def __init__(self, num_classes=40, lstm_hidden=128, lstm_layers=2,
                 dropout_rate=0.3, use_batchnorm=True):
        super(AudioResNetLSTM, self).__init__()

        self.use_bn = use_batchnorm

        # Load pretrained ResNet18 and adapt for single-channel input
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove final FC → output embedding: (batch, 512)
        self.resnet.fc = nn.Identity()

        # LSTM block
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True
        )

        # Classifier with BatchNorm + Dropout
        layers = [nn.Linear(2 * lstm_hidden, 256)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(256))
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        ])
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, 80, 117)
        x = x.unsqueeze(1)  # → (batch, 1, 80, 117)

        # ResNet backbone → (batch, 512)
        features = self.resnet(x)

        # Prepare for LSTM → (batch, 1, 512)
        lstm_input = features.unsqueeze(1)

        # LSTM output → (batch, 1, 2*lstm_hidden)
        lstm_out, _ = self.lstm(lstm_input)

        # Take last time step → (batch, 2*lstm_hidden)
        final_features = lstm_out[:, -1, :]

        # Classifier → (batch, num_classes)
        output = self.classifier(final_features)
        return output
