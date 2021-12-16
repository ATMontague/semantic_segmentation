import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from modules import eASPP


def double_conv(in_channels, out_channels):
    out = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return out


class UnetTest(nn.Module):

    def __init__(self, num_classes=1):
        super(UnetTest, self).__init__()

        self.name = 'UnetTest'

        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block1 = double_conv(3, 64)
        self.block2 = double_conv(64, 128)
        self.block3 = double_conv(128, 256)
        self.block4 = double_conv(256, 512)
        self.block5 = double_conv(512, 1024)
        self.trans_conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.block6 = double_conv(1024, 512)
        self.trans_conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.block7 = double_conv(512, 256)
        self.trans_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.block8 = double_conv(256, 128)
        self.trans_conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.block9 = double_conv(128, 64)

        self.out_conv = nn.Conv2d(in_channels=64, out_channels=self.num_classes, kernel_size=1)

        # additions
        # eASPP block after each pooling layer so that next skip connection has better features passed
        self.easpp1 = eASPP(in_channels=64, out_channels=64)
        self.easpp2 = eASPP(in_channels=128, out_channels=128)
        self.easpp3 = eASPP(in_channels=256, out_channels=256)
        self.easpp4 = eASPP(in_channels=512, out_channels=512)

    def forward(self, x):
        ###########
        # encoder #
        ###########

        skip1 = self.block1(x)
        x = self.pool(skip1)
        x = self.easpp1(x)

        skip2 = self.block2(x)
        x = self.pool(skip2)
        x = self.easpp2(x)

        skip3 = self.block3(x)
        x = self.pool(skip3)
        x = self.easpp3(x)

        skip4 = self.block4(x)
        x = self.pool(skip4)
        x = self.easpp4(x)

        x = self.block5(x)

        ###########
        # decoder #
        ###########
        x = self.trans_conv1(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.block6(x)

        x = self.trans_conv2(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.block7(x)

        x = self.trans_conv3(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.block8(x)

        x = self.trans_conv4(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.block9(x)

        x = self.out_conv(x)

        x = torch.squeeze(x)
        return x



class Unet(nn.Module):

    def __init__(self, num_classes=6):
        super(Unet, self).__init__()

        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block1 = double_conv(3, 64)
        self.block2 = double_conv(64, 128)
        self.block3 = double_conv(128, 256)
        self.block4 = double_conv(256, 512)
        self.block5 = double_conv(512, 1024)
        self.trans_conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.block6 = double_conv(1024, 512)
        self.trans_conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.block7 = double_conv(512, 256)
        self.trans_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.block8 = double_conv(256, 128)
        self.trans_conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.block9 = double_conv(128, 64)

        self.out_conv = nn.Conv2d(in_channels=64, out_channels=self.num_classes, kernel_size=1)

    def forward(self, x):

        ###########
        # encoder #
        ###########
        skip1 = self.block1(x)
        x = self.pool(skip1)

        skip2 = self.block2(x)
        x = self.pool(skip2)

        skip3 = self.block3(x)
        x = self.pool(skip3)

        skip4 = self.block4(x)
        x = self.pool(skip4)

        x = self.block5(x)

        ###########
        # decoder #
        ###########
        x = self.trans_conv1(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.block6(x)

        x = self.trans_conv2(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.block7(x)

        x = self.trans_conv3(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.block8(x)

        x = self.trans_conv4(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.block9(x)

        x = self.out_conv(x)
        return x


class UnetExpanded(nn.Module):

    def __init__(self, num_classes=6):
        super(UnetExpanded, self).__init__()

        self.num_classes = num_classes

        # first block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # second block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # third block
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 4th block
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 5th block
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
        self.conv10 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3)

        # 6th block
        self.transp1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=31, stride=1, padding=1)
        self.conv11 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

        # 7th block
        self.transp2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=55, stride=1, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3)
        self.conv14 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)

        # 8th block
        self.transp3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=103, stride=1, padding=1)
        self.conv15 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)
        self.conv16 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)

        # 9th block
        self.transp4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=199, stride=1, padding=1)
        self.conv17 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        self.conv18 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        # 1x1 convolution to produce the same number of channels as classes in dataset
        self.conv19 = nn.Conv2d(in_channels=64, out_channels=self.num_classes, kernel_size=1)

    def forward(self, x):

        # 1st
        x = F.relu(self.conv1(x))  # shape = (b, 64, 570, 570)
        x = F.relu(self.conv2(x))  # shape = (b, 64, 568, 568)
        skip1 = x.clone()
        skip1 = TF.center_crop(skip1, output_size=392)
        x = self.pool1(x)          # shape = (b, 64, 284, 284)

        # 2nd
        x = F.relu(self.conv3(x))  # shape = (b, 128, 282, 28)
        x = F.relu(self.conv4(x))  # shape = (b, 128, 280, 280)
        skip2 = x.clone()          # shape = (b, 128, 280, 280)
        skip2 = TF.center_crop(skip2, output_size=200)
        x = self.pool2(x)          # shape = (b, 128, 140, 140)

        # 3rd
        x = F.relu(self.conv5(x))  # shape = (b, 256, 138, 138)
        x = F.relu(self.conv6(x))  # shape = (b, 256, 136, 136)
        skip3 = x.clone()          # shape = (b, 256, 136, 136)
        skip3 = TF.center_crop(skip3, output_size=104)
        x = self.pool3(x)          # shape = (b, 256, 68, 68)

        # 4th
        x = F.relu(self.conv7(x))  # shape = (b, 512, 66, 66)
        x = F.relu(self.conv8(x))  # shape = (b, 512, 64, 64)
        skip4 = x.clone()          # shape = (b, 512, 64, 64)
        # crop connection to match transpose convolution
        skip4 = TF.center_crop(skip4, output_size=56)
        x = self.pool4(x)          # shape = (b, 512, 32, 32)

        # 5th
        x = F.relu(self.conv9(x))  # shape = (b, 1024, 30, 30)
        x = F.relu(self.conv10(x)) # shape = (b, 1024, 28, 28)

        # 6th
        x = self.transp1(x)               # shape = (b, 512, 56, 56)
        x = torch.cat([x, skip4], dim=1)  # shape = (b, 1024, 56, 56)
        x = F.relu(self.conv11(x))        # shape = (b, 512, 54, 54)
        x = F.relu(self.conv12(x))        # shape = (b,

        # 7th
        x = self.transp2(x)
        x = torch.cat([x, skip3], dim=1)
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))

        # 8th
        x = self.transp3(x)
        x = torch.cat([x, skip2], dim=1)
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))

        # 9th
        x = self.transp4(x)
        x = torch.cat([x, skip1], dim=1)
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))

        # output
        x = self.conv19(x)
        return x
