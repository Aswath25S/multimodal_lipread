import torch
import torch.nn as nn
import torchvision.models as models

class VGGAudioClassifier(nn.Module):
    def __init__(self, num_classes=40, version=11, dropout_rate=0.5, use_batchnorm=True):
        super(VGGAudioClassifier, self).__init__()

        self.use_bn = use_batchnorm

        # Load selected VGG variant
        self.vgg = self.get_vgg_version(version)

        # Modify first conv layer to accept 1-channel input
        self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # Adaptive pooling to ensure fixed spatial size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 3))

        # Classifier with BatchNorm + Dropout
        layers = [nn.Linear(512 * 2 * 3, 256)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(256))
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        ])
        self.vgg.classifier = nn.Sequential(*layers)

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

        # Pass through VGG feature extractor
        x = self.vgg.features(x)

        # Adaptive pooling → (batch_size, 512, 2, 3)
        x = self.adaptive_pool(x)

        # Flatten → (batch_size, 512*2*3 = 3072)
        x = torch.flatten(x, 1)

        # Classifier → (batch_size, num_classes)
        x = self.vgg.classifier(x)
        return x
