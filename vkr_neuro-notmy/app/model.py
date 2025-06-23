import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

def get_resnet50_model(num_classes=8):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class ResNet50Emotion(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNet50Emotion, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class EfficientNetEmotion(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)