import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):

    def __init__(self, num_classes=6):
        super(Unet, self).__init__()

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
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 6th block
        self.transp1 = nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=2)
        self.conv11 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

        # 7th block
        self.transp2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3)
        self.conv14 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)

        # 8th block
        self.transp3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2)
        self.conv15 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)
        self.conv16 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)

        # 9th block
        self.transp4 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2)
        self.conv17 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        self.conv18 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        # 1x1 convolution to produce the same number of channels as classes in dataset
        self.conv19 = nn.Conv2d(in_channels=64, out_channels=self.num_classes, kernel_size=1)

    def forward(self, x):

        # input shape = (b, 3, 256, 256)

        # 1st
        x = F.relu(self.conv1(x))  # shape = (b, 64, 254, 254)
        x = F.relu(self.conv2(x))  # shape = (b, 64, 252, 252)
        x = self.pool1(x)          # shape = (b, 64, 126, 126)
        skip1 = x.clone()

        # 2nd
        x = F.relu(self.conv3(x))  # shape = (b, 6, 124, 124)
        x = F.relu(self.conv4(x))  # shape = (b, 6, 122, 122)
        x = self.pool2(x)          # shape = (b, 6, 61, 61)
        skip2 = x.clone()

        # 3rd
        x = F.relu(self.conv5(x))  # shape = (b, 6, 124, 124)
        x = F.relu(self.conv6(x))  # shape = (b, 6, 122, 122)
        x = self.pool2(x)          # shape = (b, 6, 61, 61)
        skip3 = x.clone()

        # 4th
        x = F.relu(self.conv7(x))  # shape = (b,
        x = F.relu(self.conv8(x))  # shape = (b, 6
        x = self.pool2(x)          # shape = (b
        skip4 = x.clone()

        # 5th
        x = F.relu(self.conv9(x))  # shape =
        x = F.relu(self.conv10(x))  # shape = (b,
        x = self.pool2(x)          # shape = (

        # 6th
        x = self.transp1(x)
        x = torch.cat([x, skip4])
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))

        # 7th
        x = self.transp2(x)
        x = torch.cat([x, skip3])
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))

        # 8th
        x = self.transp3(x)
        x = torch.cat([x, skip2])
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))

        # 9th
        x = self.transp4(x)
        x = torch.cat([x, skip1])
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))


if __name__ == '__main__':
    net = Unet()
    print(net)
