import torch
from torch import nn as nn


def aspp_branch(k, rate):
    out = nn.Sequential(
        nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=k, dilation=rate, padding=rate),
        nn.BatchNorm2d(num_features=256)
    )
    return out


def easpp_branch(k, rate, in_channels, out_channels, final_out):
    out = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=k, dilation=rate, padding=rate),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=k, dilation=rate, padding=rate),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=final_out, kernel_size=1),
        nn.BatchNorm2d(num_features=final_out),
        nn.ReLU()
    )
    return out


class ASPP(nn.Module):
    """
    ASPP was introduced by Chen et al. in 2017
    """

    def __init__(self):
        super(ASPP, self).__init__()

        self.branch1 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        self.branch2 = aspp_branch(k=3, rate=3)
        self.branch3 = aspp_branch(k=3, rate=6)
        self.branch4 = aspp_branch(k=3, rate=12)
        self.branch5 = nn.AdaptiveAvgPool2d((24, 48))
        self.conv_1_by_1 = nn.Conv2d(in_channels=3072, out_channels=256, kernel_size=1)

    def forward(self, x):
        # assuming input to this module is [b, 2048, 24, 48]

        branch1 = self.branch1(x)  # [b, 256, 24, 48]
        branch2 = self.branch2(x)  # [b, 256, 24, 48]
        branch3 = self.branch3(x)  # [b, 256, 24, 48]
        branch4 = self.branch4(x)  # [b, 256, 24, 48]
        branch5 = self.branch5(x)  # [b, 2048, 24, 48]

        concat = torch.cat([branch1, branch2, branch3, branch4, branch5], dim=1)  # [b, 3072, 24, 48]

        x = self.conv_1_by_1(concat)  # [b, 256, 24, 48] which is same as input
        return x


class eASPP(nn.Module):
    """
    eASPP module as introduced by Valada et al. in
    "Self-Supervised Model Adaptation for Multimodal Semantic Segmentation"

    note: This version of eASPP does not include pooling layer.
    """

    # todo: modularize to add/remove branches
    def __init__(self, in_channels, out_channels):
        super(eASPP, self).__init__()

        branch_out = int(out_channels / 4)
        self.branch1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.branch2 = easpp_branch(k=3, rate=3, in_channels=in_channels, out_channels=branch_out, final_out=out_channels)
        self.branch3 = easpp_branch(k=3, rate=6, in_channels=in_channels, out_channels=branch_out, final_out=out_channels)
        self.branch4 = easpp_branch(k=3, rate=12, in_channels=in_channels, out_channels=branch_out, final_out=out_channels)
        self.conv_1_by_1 = nn.Conv2d(in_channels=int(out_channels*4), out_channels=out_channels, kernel_size=1)

    def forward(self, x):

        branch1 = self.branch1(x)  # [b, out_channels, h, w]
        branch2 = self.branch2(x)  # [b, out_channels, h, w]
        branch3 = self.branch3(x)  # [b, out_channels, h, w]
        branch4 = self.branch4(x)  # [b, out_channels, h, w]

        # print('branch1: ', branch1.shape)
        # print('branch2: ', branch2.shape)
        # print('branch3: ', branch3.shape)
        # print('branch4: ', branch4.shape)

        concat = torch.cat([branch1, branch2, branch3, branch4], dim=1)  # [b, out_channels*4, h, w]

        # print('concat: ', concat.shape)

        x = self.conv_1_by_1(concat)  # [b, out_channels, h, w] (h and w are same as input h and w)
        return x
