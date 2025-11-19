import torch
import torch.nn as nn
import torchvision.models as models

class VGGAudioClassifier(nn.Module):
    def __init__(self, num_classes=40, version=11):
        super(VGGAudioClassifier, self).__init__()

        # Load the selected VGG variant (VGG11/13/16/19 with batch norm)
        # Default VGG expects input: (batch_size, 3, H, W)
        self.vgg = self.get_vgg_version(version)

        # Modify the first convolution layer to accept 1-channel input
        # Original: Conv2d(3, 64, kernel_size=3, padding=1)
        # New expected shape: (batch_size, 1, H, W)
        self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # Replace VGG's default classifier with a custom one
        # Input to FC: flatten(512, 2, 3) → 512*2*3 = 3072
        # Final output shape: (batch_size, num_classes)
        self.vgg.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 3, 256),  # (batch_size, 3072) → (batch_size, 256)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)   # (batch_size, 256) → (batch_size, num_classes)
        )

        # Adaptive pooling ensures a fixed spatial dimension (2×3)
        # regardless of input spectrogram size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 3))

    def get_vgg_version(self, version=11):
        # Load the appropriate pretrained VGG model
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
        """
        Expected input shape:
            x: (batch_size, 80, 117)
        Represents:
            80 mel-frequency bins × 117 time frames
        """

        # Add channel dimension for convolutional layers
        # Shape becomes: (batch_size, 1, 80, 117)
        x = x.unsqueeze(1)

        # Pass spectrogram through VGG feature extractor
        # Output shape before pooling typically: (batch_size, 512, H', W')
        x = self.vgg.features(x)

        # Adaptive pooling converts any (H', W') → fixed (2, 3)
        # Resulting shape: (batch_size, 512, 2, 3)
        x = self.adaptive_pool(x)

        # Flatten spatial and channel dimensions
        # (batch_size, 512, 2, 3) → (batch_size, 512*2*3 = 3072)
        x = torch.flatten(x, 1)

        # Pass through the custom classifier
        # Output: (batch_size, num_classes)
        x = self.vgg.classifier(x)

        return x
