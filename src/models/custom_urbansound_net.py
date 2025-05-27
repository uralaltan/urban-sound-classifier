# models/custom_urbansound_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation
import math


class MultiScaleConvBlock(nn.Module):
    """Multi-scale convolution block for capturing different temporal patterns"""

    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()

        # Different kernel sizes for multi-scale feature extraction
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels // 4, 7, padding=3)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SqueezeExcitation(out_channels, out_channels // 8)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        # Multi-scale convolutions
        x1 = self.conv1x1(x)
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        x7 = self.conv7x7(x)

        # Concatenate multi-scale features
        out = torch.cat([x1, x3, x5, x7], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        out = self.se(out)
        out = self.dropout(out)

        return out


class ResidualBlock(nn.Module):
    """Residual block with pre-activation"""

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.dropout = nn.Dropout2d(dropout)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.conv2(out)

        # Add shortcut
        out += self.shortcut(residual)

        return out


class AttentionModule(nn.Module):
    """Channel and spatial attention module"""

    def __init__(self, channels):
        super().__init__()

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa

        return x


class CustomUrbanSoundNet(nn.Module):
    """
    Custom neural network designed specifically for UrbanSound8K classification
    Targets 90%+ accuracy through advanced architectural choices
    """

    def __init__(self, n_classes=10, dropout=0.2):
        super().__init__()

        # Initial feature extraction with multi-scale convolution
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Multi-scale feature extraction blocks
        self.ms_conv1 = MultiScaleConvBlock(64, 128, dropout)
        self.pool1 = nn.MaxPool2d(2)

        self.ms_conv2 = MultiScaleConvBlock(128, 256, dropout)
        self.pool2 = nn.MaxPool2d(2)

        # Residual blocks for deeper feature learning
        self.resblock1 = ResidualBlock(256, 256, dropout=dropout)
        self.resblock2 = ResidualBlock(256, 512, stride=2, dropout=dropout)
        self.resblock3 = ResidualBlock(512, 512, dropout=dropout)

        # Attention mechanism
        self.attention = AttentionModule(512)

        # Multi-scale global pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # Classification head with multiple dropouts
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),  # *2 because we concat GAP and GMP
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),

            nn.Linear(128, n_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial feature extraction
        x = self.stem(x)

        # Multi-scale convolutions
        x = self.ms_conv1(x)
        x = self.pool1(x)

        x = self.ms_conv2(x)
        x = self.pool2(x)

        # Residual blocks
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)

        # Apply attention
        x = self.attention(x)

        # Global pooling (both average and max)
        gap = self.gap(x).flatten(1)
        gmp = self.gmp(x).flatten(1)
        x = torch.cat([gap, gmp], dim=1)

        # Classification
        x = self.classifier(x)

        return x


class MixupLoss(nn.Module):
    """Mixup loss for better generalization"""

    def __init__(self, alpha=0.2, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, pred, target_a, target_b, lam):
        return lam * self.criterion(pred, target_a) + (1 - lam) * self.criterion(pred, target_b)


def mixup_data(x, y, alpha=0.2, device='cuda'):
    """Perform mixup data augmentation"""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().to(device)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


# Alternative: Even more advanced model with Transformer components
class TransformerUrbanSoundNet(nn.Module):
    """
    Transformer-enhanced model for even higher accuracy
    """

    def __init__(self, n_classes=10, d_model=512, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()

        # CNN backbone for local feature extraction
        self.backbone = CustomUrbanSoundNet(n_classes=d_model, dropout=dropout)
        # Remove the classifier from backbone
        self.backbone.classifier = nn.Identity()

        # Positional encoding for transformer
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, x):
        # Extract features using CNN backbone
        features = self.backbone(x)  # Shape: [batch, d_model]

        # Add batch and sequence dimensions for transformer
        features = features.unsqueeze(1)  # Shape: [batch, 1, d_model]

        # Add positional encoding
        features = features + self.pos_encoding[:, :features.size(1), :]

        # Apply transformer
        features = self.transformer(features)

        # Global average pooling over sequence dimension
        features = features.mean(dim=1)

        # Final classification
        return self.classifier(features)
