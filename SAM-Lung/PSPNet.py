import torch
from torch import nn
from torch.nn import functional as F

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x

class CNNAttetion(nn.Module):
    def __init__(self, in_channels):
        super(CNNAttetion, self).__init__()

        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)

        return out


class _PSPModule(nn.Module):

    def __init__(self, in_channels, bin_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)

        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, norm_layer)
                                     for b_s in bin_sizes])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), out_channels,
                      kernel_size=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )


    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):

        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)

        attention = CNNAttetion(in_channels)

        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)

        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

class PSPNetMaskDecoder(nn.Module):
    def __init__(self, num_classes, m_out_sz=256, in_channels=3):
        super(PSPNetMaskDecoder, self).__init__()

        self.m_out_sz = m_out_sz

        norm_layer = nn.SyncBatchNorm

        self.psp_module = _PSPModule(self.m_out_sz, bin_sizes=[1, 2, 3, 6], norm_layer=norm_layer)

        self.classifier = nn.Conv2d(self.m_out_sz // 4, num_classes, kernel_size=1)

        initialize_weights(self.psp_module, self.classifier)

    def forward(self, x):
        input_size = (1024, 1024)
        x = self.psp_module(x)

        output = self.classifier(x)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        output = output[:, :, :input_size[0], :input_size[1]]

        return output


