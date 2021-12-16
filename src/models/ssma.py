import torch
from torch import nn as nn
from torchvision.models import resnet50



def easpp_branch(k, rate):
    out = nn.Sequential(
        nn.Conv2d(in_channels=2048, out_channels=64, kernel_size=1),
        nn.BatchNorm2d(num_features=64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=k, dilation=rate, padding=rate),
        nn.BatchNorm2d(num_features=64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=k, dilation=rate, padding=rate),
        nn.BatchNorm2d(num_features=64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1),
        nn.BatchNorm2d(num_features=256),
        nn.ReLU()
    )
    return out


class eASPP_small(nn.Module):

    def __init__(self):
        super(eASPP_small, self).__init__()

        self.branch1 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        self.branch2 = easpp_branch(k=3, rate=3)
        self.branch3 = easpp_branch(k=3, rate=6)
        self.branch4 = easpp_branch(k=3, rate=12)
        self.conv_1_by_1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1(x)  # [b, 256, 24, 48]
        branch2 = self.branch2(x)  # [b, 256, 24, 48]
        branch3 = self.branch3(x)  # [b, 256, 24, 48]
        branch4 = self.branch4(x)  # [b, 256, 24, 48]

        concat = torch.cat([branch1, branch2, branch3, branch4], dim=1)  # [b, 1024, 24, 48]

        x = self.conv_1_by_1(concat)  # [b, 256, 24, 48] which is same as input
        return x

class Decoder(nn.Module):
    """PyTorch Module for decoder"""

    def __init__(self, C, fusion=False):
        """Constructor

        :param C: Number of categories
        :param fusion: boolean for fused skip connections (False for stage 1, True for stages 2 and 3)
        """
        super(Decoder, self).__init__()

        # variables
        self.num_categories = C
        self.fusion = fusion

        # layers stage 1
        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        nn.init.kaiming_uniform_(self.deconv1.weight, nonlinearity="relu")
        self.deconv1_bn = nn.BatchNorm2d(256)

        # layers stage 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(280, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256)
        )
        for i, layer in enumerate(self.stage2):
            if str(type(layer)) == "<class 'torch.nn.modules.conv.Conv2d'>" or \
                                   str(type(layer)) == "<class 'torch.nn.modules.conv.ConvTranspose2d'>":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

        # layers stage 3
        self.stage3 = nn.Sequential(
            nn.Conv2d(280, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.num_categories, 1),
            nn.BatchNorm2d(self.num_categories),
            nn.ConvTranspose2d(self.num_categories, self.num_categories, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(self.num_categories)
        )
        for i, layer in enumerate(self.stage3):
            if str(type(layer)) == "<class 'torch.nn.modules.conv.Conv2d'>" or \
                                   str(type(layer)) == "<class 'torch.nn.modules.conv.ConvTranspose2d'>":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

        # decoder auxiliary layers
        self.aux_conv1 = nn.Conv2d(256, self.num_categories, 1)
        nn.init.kaiming_uniform_(self.aux_conv1.weight, nonlinearity="relu")
        self.aux_conv1_bn = nn.BatchNorm2d(self.num_categories)
        self.aux_conv2 = nn.Conv2d(256, self.num_categories, 1)
        nn.init.kaiming_uniform_(self.aux_conv2.weight, nonlinearity="relu")
        self.aux_conv2_bn = nn.BatchNorm2d(self.num_categories)

        # decoder fuse skip layers
        self.fuse_conv1 = nn.Conv2d(256, 24, 1)
        nn.init.kaiming_uniform_(self.fuse_conv1.weight, nonlinearity="relu")
        self.fuse_conv1_bn = nn.BatchNorm2d(24)
        self.fuse_conv2 = nn.Conv2d(256, 24, 1)
        nn.init.kaiming_uniform_(self.fuse_conv2.weight, nonlinearity="relu")
        self.fuse_conv2_bn = nn.BatchNorm2d(24)

    def forward(self, x, skip1, skip2):
        """Forward pass

        :param x: input feature maps from eASPP
        :param skip1: skip connection 1
        :param skip2: skip connection 2
        :return: final output and auxiliary output 1 and 2
        """

        # stage 1
        x = torch.relu(self.deconv1_bn(self.deconv1(x)))
        x = torch.cat((x, skip1), 1)

        # stage 2
        x = self.stage2(x)
        x = torch.cat((x, skip2), 1)

        # stage 3
        x = self.stage3(x)
        return x

    def aux(self, x, conv, bn, scale):
        """Compute auxiliary output"""
        x = bn(conv(x))

        return nn.UpsamplingBilinear2d(scale_factor=scale)(x)

    def integrate_fuse_skip(self, x, fuse_skip, conv, bn):
        """Integrate fuse skip connection with decoder"""
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = torch.relu(bn(conv(x)))

        return torch.mul(x, fuse_skip)


class eASPP(nn.Module):
    """PyTorch Module for eASPP"""

    def __init__(self):
        """Constructor

        Initializes the 5 branches of the eASPP network.
        """

        super(eASPP, self).__init__()

        # branch 1
        self.branch1_conv = nn.Conv2d(2048, 256, kernel_size=1)
        self.branch1_bn = nn.BatchNorm2d(256)

        self.branch234 = nn.ModuleList([])
        self.branch_rates = [3, 6, 12]
        for rate in self.branch_rates:
            # branch 2
            branch = nn.Sequential(
                nn.Conv2d(2048, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, dilation=rate, padding=rate),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, dilation=rate, padding=rate),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
            self.branch234.append(branch)
        for i, sequence in enumerate(self.branch234):
            for ii, layer in enumerate(sequence):
                if str(type(layer)) == "<class 'torch.nn.modules.conv.Conv2d'>":
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

        # branch 5
        self.branch5_conv = nn.Conv2d(2048, 256, 1)
        nn.init.kaiming_uniform_(self.branch5_conv.weight, nonlinearity="relu")
        self.branch5_bn = nn.BatchNorm2d(256)

        # final layer
        self.eASPP_fin_conv = nn.Conv2d(1280, 256, kernel_size=1)
        nn.init.kaiming_uniform_(self.eASPP_fin_conv.weight, nonlinearity="relu")
        self.eASPP_fin_bn = nn.BatchNorm2d(256)

    def forward(self, x):
        """Forward pass

        :param x: input from encoder (in stage 1) or from fused encoders (in stage 2 and 3)
        :return: feature maps to be forwarded to decoder
        """
        # branch 1: 1x1 convolution
        out = torch.relu(self.branch1_bn(self.branch1_conv(x)))

        # branch 2-4: atrous pooling
        y = self.branch234[0](x)
        out = torch.cat((out, y), 1)
        y = self.branch234[1](x)
        out = torch.cat((out, y), 1)
        y = self.branch234[2](x)
        out = torch.cat((out, y), 1)

        # branch 5: image pooling
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        # print("========== x.shape: ", x.shape)
        x = torch.relu(self.branch5_bn(self.branch5_conv(x)))
        x = nn.Upsample((24, 48), mode="bilinear")(x)
        out = torch.cat((out, x), 1)

        return torch.relu(self.eASPP_fin_bn(self.eASPP_fin_conv(out)))


class Encoder(nn.Module):
    """PyTorch Module for encoder"""

    def __init__(self):
        super(Encoder, self).__init__()

        # As seen from the diagram we can see this is the last step before the
        # angled arrow. It does a 246 convolution and then uses a batch norm to downsample to 24
        self.enc_skip2_conv = nn.Conv2d(256, 24, kernel_size=1, stride=1)
        self.enc_skip2_conv_bn = nn.BatchNorm2d(24)

        # Similarly, this is the last steps of skip 1, a 512 convolution and then a batch norm to downsample
        self.enc_skip1_conv = nn.Conv2d(512, 24, kernel_size=1, stride=1)
        self.enc_skip1_conv_bn = nn.BatchNorm2d(24)

        # Sets up the ReLU for the pre-activation residual units.
        # Skip2 uses light green and Skip1 uses dark green. Both of these are from ResNet-50
        nn.init.kaiming_uniform_(self.enc_skip2_conv.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.enc_skip1_conv.weight, nonlinearity="relu")

        # Uses the https://pytorch.org/hub/pytorch_vision_resnet/
        self.res_n50_enc = resnet50(pretrained=True)
        # Replace the final index of resnet layer2 with the custom bottleneck
        # This in effect removes the final ResNet-50 downsampling and changes the stride from 2 to 1
        self.res_n50_enc.layer2[-1] = BottleneckSSMA(512, 128, 1, 2, 64, copy_from=self.res_n50_enc.layer2[-1])

        u3_sizes_short = [(256, 1, 256, 2, 1024), (256, 1, 256, 16, 1024), (256, 1, 256, 8, 1024),
                          (256, 1, 256, 4, 1024)]
        for i, x in enumerate(u3_sizes_short):
            dropout = i == 0
            # Replace the units replace the units res4c, res4d, res4e, res4f with our proposed multiscale units with rates r1 = 1 in all the units and r2 = 2,4,8,16 correspondingly
            # pg. 7 bottom left paragraph
            self.res_n50_enc.layer3[2 + i] = BottleneckSSMA(x[-1], x[0], x[1], x[3], x[2],
                                                            copy_from=self.res_n50_enc.layer3[2 + i], drop_out=dropout)

        # In addition, we replace the last three units of block four res5a,
        # res5b, res5c with the multiscale units with increasing rates in
        # both 3×3 convolutions, as (r1 = 2,r2 = 4), (r1 = 2,r2 = 8),
        # (r1 = 2,r2 = 16) correspondingly
        u3_sizes_block = [(512, 2, 512, 4, 2048), (512, 2, 512, 8, 2048), (512, 2, 512, 16, 2048)]
        for i, res in enumerate(u3_sizes_block):
            downsample = None
            if i == 0:
                downsample = self.res_n50_enc.layer4[0].downsample
                downsample[0].stride = (1, 1)

            self.res_n50_enc.layer4[i] = BottleneckSSMA(res[-1], res[0], res[1], res[3], res[2], downsample=downsample,
                                                        copy_from=self.res_n50_enc.layer4[i])

    def forward(self, x):
        x = self.res_n50_enc.conv1(x)
        x = self.res_n50_enc.bn1(x)
        x = self.res_n50_enc.relu(x)
        x = self.res_n50_enc.maxpool(x)

        x = self.res_n50_enc.layer1(x)  # this connection goes to conv and then decoder/skip 2
        s2 = self.enc_skip2_conv_bn(self.enc_skip2_conv(x))

        x = self.res_n50_enc.layer2(x)
        s1 = self.enc_skip1_conv_bn(self.enc_skip1_conv(x))

        x = self.res_n50_enc.layer3(x)

        x = self.res_n50_enc.layer4(x)

        return x, s2, s1


class BottleneckSSMA(nn.Module):
    """PyTorch Module for multi-scale units (modified residual units) for Resnet50 stages"""

    def __init__(self, inplanes, planes, r1, r2, d3, stride=1, downsample=None, copy_from=None, drop_out=False):
        """Constructur

        :param inplanes: input dimension
        :param planes: output dimension
        :param r1: dilation rate and padding 1
        :param r2: dilation rate and padding 2
        :param d3: split factor
        :param stride: stride
        :param downsample: down sample rate
        :param copy_from: copy of residual unit from second/third stage resnet50
        :param drop_out: boolean for inclusion of dropout layer
        """
        super(BottleneckSSMA, self).__init__()
        self.dropout = drop_out

        half_d3 = int(d3 / 2)

        self.conv2a = nn.Conv2d(planes, half_d3, kernel_size=3, stride=1, dilation=r1,
                                padding=r1, bias=False)
        self.bn2a = nn.BatchNorm2d(half_d3)
        self.conv2b = nn.Conv2d(planes, half_d3, kernel_size=3, stride=1, dilation=r2,
                                padding=r2, bias=False)
        self.bn2b = nn.BatchNorm2d(half_d3)
        self.conv3 = nn.Conv2d(d3, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)

        nn.init.kaiming_uniform_(self.conv2a.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2b.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity="relu")

        if copy_from is None:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
        else:
            self.conv1 = copy_from.conv1
            self.bn1 = copy_from.bn1

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward Pass

        :param x: input feature maps
        :return: output feature maps
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out_a = self.conv2a(out)
        out_a = self.bn2a(out_a)
        out_a = self.relu(out_a)

        out_b = self.conv2b(out)
        out_b = self.bn2b(out_b)
        out_b = self.relu(out_b)

        out = torch.cat((out_a, out_b), dim=1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        if self.dropout:
            m = nn.Dropout(p=0.5)
            out = m(out)

        return out


class AdapNetReduced(nn.Module):
    """PyTorch module for a reduced version of AdapNet++"""

    def __init__(self, num_classes):
        """Constructor
        :param num_classes: number of classes
        """
        super(AdapNetReduced, self).__init__()

        self.num_categories = num_classes
        self.fusion = False
        self.encoder = Encoder()
        self.eASPP = eASPP_small()
        self.decoder = Decoder(self.num_categories, self.fusion)
        self.name = 'AdapNetReduced'

    def forward(self, x):
        """Forward pass

        In the case of AdapNet++, only 1 modality is used (either the RGB-image, or the Depth-image). With 'AdapNet++
        with fusion architecture' two modalities are used (both the RGB-image and the Depth-image).

        :param mod1: modality 1
        :param mod2: modality 2
        :return: final output and auxiliary output 1 and 2
        """
        x, skip2, skip1 = self.encoder(x)
        x = self.eASPP(x)
        x = self.decoder(x, skip1, skip2)
        return x


class SSMA(nn.Module):
    """PyTorch Module for SSMA"""

    def __init__(self, features, bottleneck):
        """Constructor

        :param features: number of feature maps
        :param bottleneck: bottleneck compression rate
        """
        super(SSMA, self).__init__()
        reduce_size = int(features / bottleneck)
        double_features = int(2 * features)
        self.link = nn.Sequential(
            nn.Conv2d(double_features, reduce_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(reduce_size, double_features, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(double_features, features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features)
        )

        nn.init.kaiming_uniform_(self.link[0].weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.link[2].weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.final_conv[0].weight, nonlinearity="relu")
        self.name = 'SSMA'

    def forward(self, x1, x2):
        """Forward pass

        :param x1: input data from encoder 1
        :param x2: input data from encoder 2
        :return: Fused feature maps
        """
        x_12 = torch.cat((x1, x2), dim=1)

        x_12_est = self.link(x_12)
        x_12 = x_12 * x_12_est
        x_12 = self.final_conv(x_12)

        return x_12


if __name__ == '__main__':
    HEIGHT = 384
    WIDTH = 768
    NUM_CLASSES = 5

    model = AdapNetReduced(num_classes=NUM_CLASSES)  # 31, 513,092 params
    model = model.to(device)
    dummy_input = torch.rand(2, 3, HEIGHT, WIDTH, dtype=torch.float).to(device)
    total_params, trainable_params = num_parameters(model)
    print('total params: ', total_params)
    print('trainable params: ', trainable_params)

    output = model(dummy_input)