import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class Unet(nn.Module):

    def __init__(self, num_classes=6):
        super(Unet, self).__init__()



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

        # input shape = (b, 3, 572, 572)

        # 1st
        print('input shape: ', x.shape)
        x = F.relu(self.conv1(x))  # shape = (b, 64, 570, 570)
        print('conv1 shape: ', x.shape)
        x = F.relu(self.conv2(x))  # shape = (b, 64, 568, 568)
        print('conv2 shape: ', x.shape)
        skip1 = x.clone()
        skip1 = TF.center_crop(skip1, output_size=392)
        print('skip1 shape: ', skip1.shape)
        x = self.pool1(x)          # shape = (b, 64, 284, 284)
        print('pool1 shape: ', x.shape)

        # 2nd
        print('\nsecond')
        x = F.relu(self.conv3(x))  # shape = (b, 128, 282, 28)
        print('conv3 shape: ', x.shape)
        x = F.relu(self.conv4(x))  # shape = (b, 128, 280, 280)
        print('conv4 shape: ', x.shape)
        skip2 = x.clone()          # shape = (b, 128, 280, 280)
        skip2 = TF.center_crop(skip2, output_size=200)
        print('skip2 shape: ', skip2.shape)
        x = self.pool2(x)          # shape = (b, 128, 140, 140)
        print('pool2 shape: ', x.shape)

        # 3rd
        print('\nthird')
        x = F.relu(self.conv5(x))  # shape = (b, 256, 138, 138)
        print('conv5 shape: ', x.shape)
        x = F.relu(self.conv6(x))  # shape = (b, 256, 136, 136)
        print('conv6 shape: ', x.shape)
        skip3 = x.clone()          # shape = (b, 256, 136, 136)
        skip3 = TF.center_crop(skip3, output_size=104)
        print('skip3 shape: ', skip3.shape)
        x = self.pool3(x)          # shape = (b, 256, 68, 68)
        print('pool3 shape: ', x.shape)

        # 4th
        print('\nfourth')
        x = F.relu(self.conv7(x))  # shape = (b, 512, 66, 66)
        print('conv7 shape: ', x.shape)
        x = F.relu(self.conv8(x))  # shape = (b, 512, 64, 64)
        print('conv8 shape: ', x.shape)
        skip4 = x.clone()          # shape = (b, 512, 64, 64)
        # crop connection to match transpose convolution
        skip4 = TF.center_crop(skip4, output_size=56)
        print('4th skip shape: ', skip4.shape)
        x = self.pool4(x)          # shape = (b, 512, 32, 32)
        print('pool4 shape: ', x.shape)

        # 5th
        print('\nfifth')
        x = F.relu(self.conv9(x))  # shape = (b, 1024, 30, 30)
        print('conv9 shape: ', x.shape)
        x = F.relu(self.conv10(x)) # shape = (b, 1024, 28, 28)
        print('conv10 shape: ', x.shape)
        #x = self.pool5(x)          # shape = (b, 256, 136, 136)
        #print('pool5 shape: ', x.shape)

        # 6th
        print('\nsixth')
        x = self.transp1(x)               # shape = (b, 512, 56, 56)
        print('first transpose convolution shape: ', x.shape)
        x = torch.cat([x, skip4], dim=1)  # shape = (b, 1024, 56, 56)
        print('after concating: ', x.shape)
        x = F.relu(self.conv11(x))        # shape = (b, 512, 54, 54)
        print('conv11 shape: ', x.shape)
        x = F.relu(self.conv12(x))        # shape = (b,
        print('conv11 shape: ', x.shape)

        # 7th
        print('\nseventh')
        x = self.transp2(x)
        print('2nd transpose convolution shape: ', x.shape)
        x = torch.cat([x, skip3], dim=1)
        x = F.relu(self.conv13(x))
        print('conv13 shape: ', x.shape)
        x = F.relu(self.conv14(x))
        print('conv14 shape: ', x.shape)

        # 8th
        print('\neighth')
        x = self.transp3(x)
        print('3rd transpose convolution shape: ', x.shape)
        x = torch.cat([x, skip2], dim=1)
        x = F.relu(self.conv15(x))
        print('conv15 shape: ', x.shape)
        x = F.relu(self.conv16(x))
        print('conv16 shape: ', x.shape)

        # 9th
        print('\nninth')
        x = self.transp4(x)
        print('4th transpose convolution shape: ', x.shape)
        x = torch.cat([x, skip1], dim=1)
        x = F.relu(self.conv17(x))
        print('conv17 shape: ', x.shape)
        x = F.relu(self.conv18(x))
        print('conv17 shape: ', x.shape)

        # output
        x = self.conv19(x)
        return x


if __name__ == '__main__':
    net = Unet()
    print(net)
