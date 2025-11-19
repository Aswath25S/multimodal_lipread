import torch
import torch.nn as nn
import torchvision.models as models

class VGGWithLSTMClassifier(nn.Module):
    def __init__(self, num_classes=40, lstm_hidden_size=128, lstm_layers=2, version=11):
        super(VGGWithLSTMClassifier, self).__init__()

        # Load selected VGG model (VGG11/VGG13/VGG16/VGG19 with BatchNorm)
        # Default VGG input shape: (batch_size, 3, H, W)
        vgg = self.get_vgg_version(version)

        # Modify first conv layer to accept 1-channel spectrograms
        # New expected input: (batch_size, 1, H, W)
        vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # Extract only the convolutional feature extractor
        self.vgg_features = vgg.features

        # Adaptive pooling to collapse frequency dimension while keeping time dimension
        # Output becomes: (batch_size, 512, time_steps, 1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 1))

        # Number of output channels from final VGG conv block
        # Feature shape before LSTM will be (batch_size, time_steps, 512)
        self.cnn_output_dim = 512

        # Bidirectional LSTM over temporal dimension
        # Input: (batch, time_steps, 512)
        # Output: (batch, time_steps, 2 * hidden_size)
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True
        )

        # Final classifier after taking last LSTM time-step
        # Input to FC: (batch_size, 2 * hidden_size)
        # Output: (batch_size, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def get_vgg_version(self, version=11):
        # Select appropriate pretrained VGG architecture
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
        Where:
            80  → mel-frequency bins
            117 → time frames
        """

        # Add channel dimension → (batch_size, 1, 80, 117)
        x = x.unsqueeze(1)

        # Pass through VGG convolutional feature extractor
        # Output shape: (batch_size, 512, H', W')
        x = self.vgg_features(x)

        # Adaptive pooling:
        # Collapse frequency dimension → 1
        # Preserve time dimension → becomes time_steps'
        # New shape: (batch_size, 512, time_steps', 1)
        x = self.adaptive_pool(x)

        # Remove last dimension and rearrange to (batch, time, features)
        # Result: (batch_size, time_steps', 512)
        x = x.squeeze(-1).permute(0, 2, 1)

        # Bidirectional LSTM processes sequence over time
        # lstm_out: (batch_size, time_steps', 2 * hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take output from the last time step
        # Shape: (batch_size, 2 * hidden_size)
        out = lstm_out[:, -1, :]

        # Classification head → final output: (batch_size, num_classes)
        return self.classifier(out)
