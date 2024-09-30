import torch.nn as nn
import torchvision.models as models

class ResNetModel(nn.Module):
    def __init__(self, num_classes, n_channels):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(weights=None)  # Load ResNet without pre-trained weights
        self.resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust input channels
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # Modify FC layer for num_classes

    def forward(self, x):
        return self.resnet(x)