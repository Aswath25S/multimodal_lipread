import torch
import torch.nn as nn
import torchvision.models as models

class AudioResNet(nn.Module):
    def __init__(self, num_classes=40):
        super(AudioResNet, self).__init__()
        
        # Load pretrained ResNet18
        # Default ResNet18 expects input shape: (batch_size, 3, H, W)
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify first conv layer to accept single-channel input instead of RGB (3 channels)
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final fully connected layer with a custom classifier
        num_features = self.resnet.fc.in_features  # usually 512 for ResNet18
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # final output: (batch_size, num_classes)
        )

    def forward(self, x):
        # Expected input shape before unsqueeze: (batch_size, time_steps, mel_bins)
        # Example: (batch_size, 80, 117)
        
        # Add a channel dimension for convolutional layers
        # Shape becomes: (batch_size, 1, time_steps, mel_bins)
        # Example: (batch_size, 1, 80, 117)
        x = x.unsqueeze(1)
        
        # Pass through ResNet
        # ResNet internal conv layers expect 4D input: (B, C, H, W)
        # Output shape: (batch_size, num_classes)
        return self.resnet(x)


# ===============================
# 🔍 Summary of Dimensions
# ===============================
# Input (single example):           (80, 117)
# Input (batched):                  (batch_size, 80, 117)
# After unsqueeze:                  (batch_size, 1, 80, 117)
# After ResNet backbone:            (batch_size, num_features)   # num_features ≈ 512 for ResNet18
# After final FC layers:            (batch_size, num_classes)    # e.g., (batch_size, 40)
# ===============================
# ResNet treats the spectrogram as a grayscale image (1-channel)
# ===============================
