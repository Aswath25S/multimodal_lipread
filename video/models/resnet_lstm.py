"""
2DCNN+BiLSTM model for visual speech recognition.
Based on ResNet-18 architecture combined with BiLSTM for temporal processing.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent.parent))


class TimeDistributed(nn.Module):
    """
    Apply the same module to each temporal slice of input.
    Similar to Keras TimeDistributed wrapper.
    """
    
    def __init__(self, module):
        """
        Initialize TimeDistributed wrapper.
        
        Args:
            module (nn.Module): Module to apply to each time step
        """
        super(TimeDistributed, self).__init__()
        self.module = module
    
    def forward(self, x):
        """
        Apply module to each temporal slice of input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, T, F)
        """
        batch_size, channels, time_steps, height, width = x.size()
        
        # Reshape input to (B*T, C, H, W)
        x_reshaped = x.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * time_steps, 
                                                               channels, height, width)
        
        # Apply the module
        output = self.module(x_reshaped)
        
        # Reshape output back to (B, T, F)
        output = output.view(batch_size, time_steps, -1)
        
        return output


class ResNet2DBiLSTM(nn.Module):
    """
    2DCNN+BiLSTM model for visual speech recognition.
    Uses ResNet-18 as the backbone CNN and BiLSTM for temporal processing.
    """
    
    def __init__(self, num_classes, config):
        """
        Initialize the ResNet2DBiLSTM model.
        
        Args:
            num_classes (int): Number of output classes
            config (Dict[str, Any]): Model configuration
        """
        super(ResNet2DBiLSTM, self).__init__()
        
        # Get configuration parameters
        resnet_version = config.get('model.resnet_version', 18)
        feature_dim = config.get('model.feature_dim', 1024)
        fc_hidden_size = config.get('model.fc_hidden_size', 1024)
        dropout = config.get('model.dropout', 0.5)
        
        # Load the appropriate ResNet model
        if resnet_version == 18:
            base_model = models.resnet18(pretrained=True)
        elif resnet_version == 34:
            base_model = models.resnet34(pretrained=True)
        elif resnet_version == 50:
            base_model = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unsupported ResNet version: {resnet_version}")
        
        # Modify the first convolutional layer to accept the input size
        # The original expects 224x224 images, but our input is 44x44
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final fully connected layer and pooling
        self.cnn_features = nn.Sequential(*list(base_model.children())[:-2])
        
        # Add a global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Calculate CNN output feature dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 44, 44)
            output = self.cnn_features(dummy_input)
            output = self.global_pool(output)
            cnn_output_dim = output.view(-1).size(0)
        
        # Create time-distributed CNN
        self.time_distributed_cnn = TimeDistributed(nn.Sequential(
            self.cnn_features,
            self.global_pool,
            nn.Flatten()
        ))
        
        # BiLSTM for temporal processing
        self.bilstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=feature_dim // 2,  # Bidirectional will double this
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # ReLU activation
        self.relu = nn.ReLU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Final fully connected layer for classification
        self.fc = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes)
        """
        # Apply time-distributed CNN to extract features from each frame
        # Input: (B, C, T, H, W) -> Output: (B, T, F)
        x = self.time_distributed_cnn(x)
        
        # Apply BiLSTM to process temporal features
        # Input: (B, T, F) -> Output: (B, T, F')
        x, _ = self.bilstm(x)
        
        # Take the final output of the BiLSTM
        # Input: (B, T, F') -> Output: (B, F')
        x = x[:, -1, :]
        
        # Apply ReLU activation
        x = self.relu(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply final fully connected layer
        # Input: (B, F') -> Output: (B, num_classes)
        x = self.fc(x)
        
        return x


def create_model(num_classes, config):
    """
    Create a 2DCNN+BiLSTM model for visual speech recognition.
    
    Args:
        num_classes (int): Number of output classes
        config (Dict[str, Any]): Model configuration
        
    Returns:
        nn.Module: Model instance
    """
    return ResNet2DBiLSTM(num_classes, config)
