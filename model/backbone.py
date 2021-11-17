from torch import nn
from torchvision import models


class VGG(nn.Module):
    def __init__(self, dropout=0.5, hidden=256, pretrained=True):
        super(VGG, self).__init__()
        model = models.vgg16_bn(pretrained=pretrained)

        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.MaxPool2d):
                model.features[i] = nn.AvgPool2d(kernel_size=2, stride=2)

        self.features = model.features
        self.dropout = nn.Dropout(0.5)
        self.conv1 = nn.Conv2d(512, hidden, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        x = self.conv1(x)

        x = x.tranpose(-1, -2)
        x = x.flatten(2)
        x = x.permute(-1, 0, 1)
        return x


def freeze(model):
    for name, param in model.features.named_parameters():
        if name != 'conv1':
            param.requires_grad = False


def unfreeze(model):
    for param in model.features.parameters():
        param.requires_grad = True