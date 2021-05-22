import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


def model():
    # use pretrained ResNet-18
    pretrained_net = torchvision.models.resnet18(pretrained=True)
    # remove last two layers since they aren't fully convolutional layers
    net = nn.Sequential(*list(pretrained_net.children())[:-2])
    print(net)


if __name__ == '__main__':
    model()
