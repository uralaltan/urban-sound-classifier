import torch.nn as nn
from torch.hub import load as load_hub


class TransferACDNet(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.backbone = load_hub('qiuqiangkong/panns_audio', 'cnn14', pretrained=True)
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
