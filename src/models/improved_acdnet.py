import torch
import torch.nn as nn
from torchvision.ops import SqueezeExcitation


class ImprovedACDNet(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()

        def conv_block(cin, cout, p=0.25):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout, momentum=0.95),
                nn.ReLU(),
                SqueezeExcitation(cout, cout // 8),
                nn.Dropout(p)
            )

        self.layer1 = conv_block(1, 32)
        self.branch3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.branch5 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn_ms = nn.BatchNorm2d(128, momentum=0.95)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        x = self.relu(self.bn_ms(torch.cat([b3, b5], dim=1)))
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return self.classifier(x)
