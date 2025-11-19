import torch
import torch.nn as nn
import torchvision.models as models

class LSTMResNet(nn.Module):
    def __init__(self, num_classes=40, input_size=117, dropout_rate=0.3, use_batchnorm=True):
        super(LSTMResNet, self).__init__()

        self.use_bn = use_batchnorm

        # Step 1: First BiLSTM across mel-frequency rows
        self.initial_bilstm = nn.LSTM(
            input_size=input_size,  # 117
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Step 2: ResNet18 feature extractor
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()  # keep 512-dim output

        # Step 3: Fully connected projection with BatchNorm + Dropout
        layers = [nn.Linear(512, 256)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(256))
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        self.fc = nn.Sequential(*layers)

        # Step 4: Second BiLSTM
        self.final_bilstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Step 5: Final classifier
        self.classifier = nn.Linear(2 * 128, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Step 1: First BiLSTM per mel row
        x1 = x.view(batch_size * 80, 117).unsqueeze(1)
        x1, _ = self.initial_bilstm(x1)

        # Reshape back to image-like form for ResNet
        x1 = x1.squeeze(1).view(batch_size, 1, 80, -1)

        # Step 2: ResNet feature extraction
        resnet_out = self.resnet(x1)

        # Step 3: Dense projection
        fc_out = self.fc(resnet_out)

        # Step 4: Expand features for second LSTM
        seq_input = fc_out.unsqueeze(1)
        lstm_out, _ = self.final_bilstm(seq_input)

        # Step 5: Take last time step
        final_out = lstm_out[:, -1, :]

        # Step 6: Final classification
        return self.classifier(final_out)
