import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.deconv = nn.ConvTranspose2d(in_channels=6, out_channels=6, kernel_size=4, stride=2)  # attempt at 'upsampling' to return to original img size

    def forward(self, x):
        x = F.relu(self.conv1(x))  # shape = (b, 3, 250, 250)
        x = self.pool(x)           # shape = (b, 6, 124, 124)
        x = self.deconv(x)         # shape = (b, 6, 250, 250)
        return x


if __name__ == '__main__':
    net = Net()
    print(net)