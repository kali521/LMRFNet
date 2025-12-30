# coding=utf-8
# Version:python 3.7

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import nn
from kb_utils import LayerNorm2d, SimpleGate


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, downsample=False):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channel, out_channel, 3, 2, 1) if downsample \
            else nn.Conv2d(in_channel, out_channel, 1, 1)
        self.conv1 = nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv0(x)
        return self.conv2(self.relu(self.conv1(x))) + x


# Convolution layer
def Conv2d(input, output, k=3, s=1, dilation=1, activate='relu', group=1):
    model = []
    # dilation: dilation rate of dilated convolutions
    model.append(nn.Conv2d(input, output, kernel_size=k, stride=s, dilation=dilation, padding=((k - 1) * dilation // 2),
                           groups=group))
    if activate == 'relu':
        model.append(nn.LeakyReLU(0.2, inplace=True))
    if activate == 'tanh':
        model.append(nn.Tanh())

    return nn.Sequential(*model)


# Initial layer
class DownSample(nn.Module):
    def __init__(self, scale, channel):
        super(DownSample, self).__init__()
        self.scale = scale
        self.downsample = nn.AvgPool2d(scale)
        self.conv = Conv2d(channel, channel * scale, activate=None)

    def forward(self, x):
        out = self.downsample(x)
        out = self.conv(out)
        return out


class Initial_Layer(nn.Module):
    # middle = [64, 128, 256]
    def __init__(self, input, middle):
        super(Initial_Layer, self).__init__()
        # Original scale
        self.conv1_1 = Conv2d(input, middle[0], activate=None)
        self.conv1_2 = Conv2d(middle[0], middle[0], activate=None)
        # Downsampling-Intermediate Scale
        self.down2_1 = DownSample(2, input)
        self.conv2_2 = Conv2d(input*2, middle[1], activate=None)
        # Downsampling-low scale
        self.down3_1 = DownSample(4, input)
        self.conv3_2 = Conv2d(input*4, middle[2], activate=None)

    def forward(self, x):
        in_x = x
        out1 = self.conv1_2(self.conv1_1(in_x))
        out2 = self.conv2_2(self.down2_1(in_x))
        out3 = self.conv3_2(self.down3_1(in_x))

        return out1, out2, out3


# Coarse-fusion network
class UpSample(nn.Module):
    # Given input channel and scale, upsample the input x to shape_size
    def __init__(self, scale, channel):
        super(UpSample, self).__init__()
        self.transpose = nn.Sequential(
            nn.ConvTranspose2d(channel, channel // scale, kernel_size=scale, stride=scale))

    def forward(self, x, shape_size):
        yh, yw = shape_size
        out = self.transpose(x)
        xh, xw = out.size()[-2:]
        h = yh - xh
        w = yw - xw
        pt = h // 2
        pb = h - pt
        pl = w // 2
        pr = w - w // 2
        out = F.pad(out, (pl, pr, pt, pb), mode='reflect')
        return out


class AdaIN(nn.Module):
    # input: number of channels in the noise estimation map, guidance information
    # channel: input x
    def __init__(self, down, input, channel):
        super(AdaIN, self).__init__()
        self.down = down
        if down:
            self.up = UpSample(down,input)
        self.con1 = Conv2d(input//down, channel, k=3, activate=None)
        self.con1_f = Conv2d(channel, channel, k=3, activate=None)
        self.con2 = Conv2d(channel, channel, activate=None)
        self.con3 = Conv2d(channel, channel, activate=None)

    def forward(self, de_map, guide_info):
        _, c, h, w = de_map.size()
        if self.down:
            guide_info = self.up(guide_info, de_map.size()[-2:] )
        # Normalize
        mu = torch.mean(de_map.view(-1, c, h * w), dim=2)[:, :, None, None]
        sigma = torch.std(de_map.view(-1, c, h * w), dim=2)[:, :, None, None] + 10e-5
        de_map = (de_map - mu) / sigma

        guide_info = self.con1(guide_info)
        guide_info = self.con1_f(guide_info)
        gama = self.con2(guide_info)
        beta = self.con3(guide_info)   

        de_map = de_map * gama + beta

        return de_map


class AdaINBlock(nn.Module):
    def __init__(self, down, input, channel):
        super(AdaINBlock, self).__init__()
        self.con = Conv2d(channel, channel, activate=None)
        self.adain1 = AdaIN(down, input, channel)
        self.con1 = Conv2d(channel, channel, activate=None)

    def forward(self, demap, est_noise):
        x = self.con(demap)
        x = self.adain1(x, est_noise)
        x = self.con1(x)

        return demap + x


class AdaMScaleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # self.norm1 = LayerNorm2d(in_channel)
        # self.norm2 = LayerNorm2d(in_channel)
        self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.conv4 = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(out_channel, out_channel), nn.Sigmoid())
        # Spatial attention
        self.conv = nn.Sequential(nn.Conv2d(1, 1, 7, 1, 3), nn.Sigmoid())
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # x = self.norm1(x)  # Channel number normalization
        ch_att =  self.fc(self.avg_pool(x).squeeze())\
            .unsqueeze(dim=-1).unsqueeze(dim=-1)
        out_ch = ch_att*x
        sp_att = self.conv(torch.mean(x, dim=1, keepdim=True))
        out_sp = sp_att*x
        x = out_ch + out_sp 
        x = self.conv3(x)
        # FFN
        # x_F = self.norm2(x)
        x_F = self.conv4(x)
        x_F = self.conv5(x_F)
        return x + x_F    


# Concatenation and split
class CoAttention(nn.Module):
    def __init__(self, scale, channel, ratio):
        super(CoAttention, self).__init__()
        self.scale = scale

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.sq = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.LeakyReLU(True),
            nn.Linear(channel // ratio, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        out = self.pooling(x).view(b, c)
        out = self.sq(out).view(b, c, 1, 1)
        out = out * x
        out = torch.sum(out.view(b, self.scale, c // self.scale, h, w), dim=1, keepdim=False)
        return out


class CASC(nn.Module):
    def __init__(self, scale, inchannels, ratio=4,stride=1,dilation=1,cardinality=1,padding=1,groups=1,):
        super(CASC, self).__init__()
        self.CA = CoAttention(scale, inchannels * scale, ratio)
        self.scale = scale
        pooling_r = 4
        # k1: Through convolution K1
        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        inchannels, inchannels, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        groups=cardinality, bias=False),
                    )
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r), 
                    nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=inchannels, bias=False),
                    )
        # k3: Through convolution K3 (lower branch)
        self.k3 = nn.Sequential(
                    nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    )
        # k4: Through convolution K4 (lower branch)
        self.k4 = nn.Sequential(
                    nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    )
        
    def forward(self, x):
        b, c, h, w = x.size()
        out = x.view(b, self.scale, c // self.scale, h, w)  # Reshape tensor
        # Directly extract each part through slicing
        x1 = out[:, 0, :, :, :]  # Get first part (upper)
        x2 = out[:, 1, :, :, :]  # Get second part (middle)
        x3 = out[:, 2, :, :, :]  # Get third part (lower guidance)
        identity = x3
        # Upper branch: input feature x through k2, upsample to input size, then residual connection with input
        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x3), identity.size()[2:]))) # sigmoid(identity + k2)
        # Lower branch: input feature x through k3, then matrix multiplication with upper branch output
        out = torch.mul(self.k3(x3), out) # k3 * sigmoid(identity + k2)
        # Finally, pass output through k4
        out3 = self.k4(out) # k4
        out2 = self.k1(x2)
        out_concat = torch.cat([x1,out2,out3],dim=1)
        fm = self.CA(out_concat)
        return  fm


class MSCASC(nn.Module):
    def __init__(self, n_scale, middle, ratio=4):
        super(MSCASC, self).__init__()
        self.n_scale = n_scale
        self.sample_dict = nn.ModuleDict()

        self.dm = nn.ModuleList([CASC(n_scale, middle[i], ratio) for i in range(n_scale)])

        for i in range(n_scale):
            for j in range(n_scale):
                if i < j:
                    self.sample_dict.update({f'{i + 1}_{j + 1}': DownSample(2 ** (j - i), middle[i])})
                if i > j:
                    self.sample_dict.update({f'{i + 1}_{j + 1}': UpSample(2 ** (i - j), middle[i])})

    def select_sample(self, x, shape_size, i, j):
        if i == j:
            return x
        else:
            if i > j:
                return self.sample_dict[f'{i + 1}_{j + 1}'](x, shape_size)
            else:
                return self.sample_dict[f'{i + 1}_{j + 1}'](x)

    def forward(self, x):
        res = []
        for i in range(self.n_scale):
            # Scale normalization
            fuse = [self.select_sample(x[j], x[i].size()[-2:], j, i) for j in range(self.n_scale)]
            # Co-attention
            res.append(self.dm[i](torch.cat(fuse, dim=1)))

        return res


# Main network
class LMRFNet(nn.Module):
    def __init__(self, input=1, output=1,n_channel=32,middle=[32, 64, 128],ratio=4):
        super(LMRFNet, self).__init__()
        ### SFE: Shallow Feature Extraction
        self.conv1 = ResBlock(input, n_channel, downsample=False)
        self.conv2 = ResBlock(n_channel, n_channel*2, downsample=True)
        self.conv3 = ResBlock(n_channel*2, n_channel*4, downsample=True)

        ### MSFI: Multi-Scale Feature Interaction
        self.adain1_1 = AdaINBlock(2, middle[1], middle[0])
        self.adain1_2 = AdaINBlock(2, middle[1], middle[0])

        self.adain2_1 = AdaINBlock(2, middle[2], middle[1])
        self.adain2_2 = AdaINBlock(2, middle[2], middle[1])

        self.con3_1 = Conv2d(middle[2], middle[2], activate=None)
        self.con3_2 = Conv2d(middle[2], middle[2], activate=None)

        ### HAT: Hierarchical Attention Transformer
        # Multi-scale adaptive nets
        self.hat1 = AdaMScaleBlock(n_channel, n_channel)
        self.hat2 = AdaMScaleBlock(n_channel*2, n_channel*2)
        self.hat3 = AdaMScaleBlock(n_channel*4, n_channel*4)

        ### AFF+reconstruction
        self.msfda1 = MSCASC(3, middle, ratio)
        self.msfda2 = MSCASC(3, middle, ratio)
        self.fda = CASC(3, middle[0])
        self.up2_1 = UpSample(2, middle[1])
        self.up3_1 = UpSample(4, middle[2])
        self.out = Conv2d(middle[0], output, activate=None)

    def forward(self, x):
        # Shallow Feature Extraction
        SFE_1 = self.conv1(x)
        SFE_2 = self.conv2(SFE_1)
        SFE_3 = self.conv3(SFE_2)

        # Multi-Scale Feature Interaction
        MSFI1_1 = self.adain1_1(SFE_1, SFE_2)
        MSFI2_1 = self.adain2_1(SFE_2, SFE_3)
        conv3_1 = self.con3_1(SFE_3)

        MSFI1_2 = self.adain1_2(MSFI1_1, MSFI2_1)
        MSFI2_2 = self.adain2_2(MSFI2_1, conv3_1)
        conv3_2 = self.con3_2(conv3_1)

        # Hierarchical Attention Transformer
        HAT_1 = self.hat1(MSFI1_2)
        HAT_2 = self.hat2(MSFI2_2)
        HAT_3 = self.hat3(conv3_2)
        
        # Adaptive Feature Fusion
        AFF1_1, AFF2_1, AFF3_1 = self.msfda1([HAT_1, HAT_2, HAT_3])
        AFF1_2, AFF2_2, AFF3_2 = self.msfda2([AFF1_1, AFF2_1, AFF3_1])

        # Upsampling
        AFF2_3 = self.up2_1(AFF2_2, AFF1_2.size()[-2:])
        AFF3_3 = self.up3_1(AFF3_2, AFF1_2.size()[-2:])

        # Final fusion
        Fine_fusion = self.fda(torch.cat([AFF1_2, AFF2_3, AFF3_3], dim=1))

        # Output
        out = self.out(Fine_fusion)
        X = x - out
        return X