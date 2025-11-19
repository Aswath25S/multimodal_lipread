import torch
import torch.nn as nn
import torchvision.models as models

class AudioResNet(nn.Module):
    def __init__(self, num_classes=40, dropout_rate=0.5, use_batchnorm=True):
        super(AudioResNet, self).__init__()

        self.use_bn = use_batchnorm

        # Load pretrained ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify first conv layer for single-channel (grayscale) spectrogram input
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace final FC layer with classifier including BatchNorm + Dropout
        num_features = self.resnet.fc.in_features

        layers = [
            nn.Linear(num_features, 512),
        ]

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(512))

        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        ])

        self.resnet.fc = nn.Sequential(*layers)

    def forward(self, x):
        # Input expected: (batch_size, time_steps, mel_bins)
        x = x.unsqueeze(1)  # -> (batch_size, 1, time_steps, mel_bins)
        return self.resnet(x)