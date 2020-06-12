# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class FootprintNetwork(nn.Module):

    def __init__(self, pretrained=True):
        super(FootprintNetwork, self).__init__()
        self.encoder = ResnetEncoder(pretrained=pretrained)
        self.mask_decoder = SkipDecoder()
        self.depth_decoder = SkipDecoder()

    def forward(self, input_image):
        features = self.encoder(input_image)
        mask_outputs = self.mask_decoder(features)
        depth_outputs = self.depth_decoder(features)

        outputs = {}
        for key in mask_outputs:
            outputs[key] = torch.cat([mask_outputs[key], depth_outputs[key]], dim=1)

        return outputs


class ResnetEncoder(nn.Module):

    def __init__(self, pretrained=True):
        super(ResnetEncoder, self).__init__()

        encoder = resnet34(pretrained=pretrained)

        self.layer0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        del encoder

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225

        x = self.layer0(x)
        self.features.append(x)
        self.features.append(self.layer1(self.features[-1]))
        self.features.append(self.layer2(self.features[-1]))
        self.features.append(self.layer3(self.features[-1]))
        self.features.append(self.layer4(self.features[-1]))

        return self.features


class SkipDecoder(nn.Module):

    def __init__(self):
        super(SkipDecoder, self).__init__()

        self.block1 = ConvUpsampleAndConcatBlock(in_ch=512, out_ch=256, use_elu=True, use_bn=False)
        self.block2 = ConvUpsampleAndConcatBlock(in_ch=256, out_ch=128, use_elu=True, use_bn=False)
        self.block3 = ConvUpsampleAndConcatBlock(in_ch=128, out_ch=64, use_elu=True, use_bn=False)
        self.block4 = ConvUpsampleAndConcatBlock(in_ch=64, out_ch=64, use_elu=True, use_bn=False)

        self.outconv1 = OutConvBlock(in_ch=128, out_ch=2, scale=8)
        self.outconv2 = OutConvBlock(in_ch=64, out_ch=2, scale=4)
        self.outconv3 = OutConvBlock(in_ch=64, out_ch=2, scale=2)

        self.outconv4 = nn.Sequential(ConvBlock(in_ch=64, out_ch=32, use_elu=True, use_bn=False),
                                      OutConvBlock(in_ch=32, out_ch=2, scale=1))

    def forward(self, features):

        outputs = {}
        x = features[-1]

        x = self.block1(x, features[-2])

        x = self.block2(x, features[-3])
        outputs['1/8'] = self.outconv1(x)

        x = self.block3(x, features[-4])
        outputs['1/4'] = self.outconv2(x)

        x = self.block4(x, features[-5])
        outputs['1/2'] = self.outconv3(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        outputs['1/1'] = self.outconv4(x)

        return outputs


class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, use_elu=True, use_bn=False):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.pad = nn.ReflectionPad2d(1)

        if use_elu:
            self.non_lin = nn.ELU(inplace=True)
        else:
            self.non_lin = nn.ReLU(inplace=True)

        self.use_bn = use_bn

    def forward(self, x):

        x = self.pad(x)
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.non_lin(x)

        x = self.pad(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.non_lin(x)

        return x


class ConvUpsampleAndConcatBlock(nn.Module):

    def __init__(self, in_ch, out_ch, use_elu=True, use_bn=False):
        super(ConvUpsampleAndConcatBlock, self).__init__()

        self.pre_concat_conv = ConvBlock(in_ch=in_ch, out_ch=out_ch, use_elu=use_elu,
                                         use_bn=use_bn)
        self.post_concat_conv = ConvBlock(in_ch=out_ch*2, out_ch=out_ch, use_elu=use_elu,
                                          use_bn=use_bn)

    def forward(self, x, cat_feats):

        x = self.pre_concat_conv(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, cat_feats], 1)
        x = self.post_concat_conv(x)

        return x


class OutConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, scale):
        super(OutConvBlock, self).__init__()

        self.pad = nn.ReflectionPad2d(1)
        self.scale = scale
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.sigmoid(x)

        if self.scale != 1:
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)

        return x


class AvgPoolConv(nn.Module):
    def __init__(self, in_ch, out_ch, size):
        super(AvgPoolConv, self).__init__()

        self.size = size
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, size=self.size, mode='nearest')

        return x
