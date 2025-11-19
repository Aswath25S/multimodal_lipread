import torch
import torch.nn as nn
import torchvision.models as models

class VGGWithLSTMClassifier(nn.Module):
    def __init__(self, num_classes=40, lstm_hidden_size=128, lstm_layers=2,
                 version=11, dropout_rate=0.3, use_batchnorm=True):
        super(VGGWithLSTMClassifier, self).__init__()

        self.use_bn = use_batchnorm

        # Load selected VGG model
        vgg = self.get_vgg_version(version)

        # Modify first conv layer for single-channel spectrogram input
        vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.vgg_features = vgg.features

        # Adaptive pooling: collapse frequency dimension
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.cnn_output_dim = 512

        # Bidirectional LSTM over temporal dimension
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True
        )

        # Classifier with BatchNorm + Dropout
        layers = [nn.Linear(2 * lstm_hidden_size, 128)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(128))
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        ])
        self.classifier = nn.Sequential(*layers)

    def get_vgg_version(self, version=11):
        if version == 11:
            return models.vgg11_bn(weights=models.VGG11_BN_Weights.DEFAULT)
        elif version == 13:
            return models.vgg13_bn(weights=models.VGG13_BN_Weights.DEFAULT)
        elif version == 16:
            return models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        elif version == 19:
            return models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
        else:
            raise ValueError(f"Invalid VGG version: {version}")

    def forward(self, x):
        # Input: (batch_size, 80, 117)
        x = x.unsqueeze(1)  # → (batch_size, 1, 80, 117)

        # VGG feature extractor
        x = self.vgg_features(x)  # → (batch_size, 512, H', W')

        # Adaptive pooling → (batch_size, 512, time_steps', 1)
        x = self.adaptive_pool(x)

        # Rearrange for LSTM → (batch_size, time_steps', 512)
        x = x.squeeze(-1).permute(0, 2, 1)

        # LSTM → (batch_size, time_steps', 2*lstm_hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take last time step → (batch_size, 2*lstm_hidden_size)
        out = lstm_out[:, -1, :]

        # Classifier → (batch_size, num_classes)
        return self.classifier(out)
