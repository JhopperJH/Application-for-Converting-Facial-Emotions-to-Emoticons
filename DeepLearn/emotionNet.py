import torch.nn as nn
import torch.nn.functional as F

class EmotionNet(nn.Module):
    network_config = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

    def __init__(self, num_of_channels, num_of_classes):
        super(EmotionNet, self).__init__()
        self.num_of_channels = num_of_channels
        self.features = self._make_layers(num_of_channels, self.network_config)
        self.classifier = nn.Sequential(nn.Linear(6*6*128, 64), nn.ELU(True), nn.Dropout(p=0.5), nn.Linear(64, num_of_classes))

    def forward(self, X):
        out = self.features(X)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=True)
        out = self.classifier(out)
        return out

    def _make_layers(self, in_channels, cfg):
        layers = []
        for X in cfg:
            if X == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, X, kernel_size=3, padding=1), nn.BatchNorm2d(X), nn.ELU(inplace=True)]
                in_channels = X
        return nn.Sequential(*layers)