# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_1(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 7, padding=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 7, padding=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv_1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv_1, self).__init__()
        self.conv = double_conv_1(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class down_convstride_2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_convstride_2, self).__init__()
        self.mpconv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch-out_ch,in_ch-out_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        #print('deconvolution:',x1.size())
        diffX = x2.size()[2] - x1.size()[2]
        if diffX % 2 != 0:
            ch1, ch2 = int(diffX/2), int(diffX/2) + 1
        else:
            ch1, ch2 = int(diffX/2), int(diffX/2)

        diffY = x2.size()[3] - x1.size()[3]
        if diffY % 2 != 0:
            cw1, cw2 = int(diffY/2), int(diffY/2) + 1
        else:
            cw1, cw2 = int(diffY/2), int(diffY/2)

        x1 = F.pad(x1, (cw1,cw2,ch1,ch2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
class up_only(nn.Module):
    def __init__(self):
        super(up_only, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        #print('deconvolution:',x1.size())
        diffX = x2.size()[2] - x1.size()[2]
        if diffX % 2 != 0:
            ch1, ch2 = int(diffX/2), int(diffX/2) + 1
        else:
            ch1, ch2 = int(diffX/2), int(diffX/2)

        diffY = x2.size()[3] - x1.size()[3]
        if diffY % 2 != 0:
            cw1, cw2 = int(diffY/2), int(diffY/2) + 1
        else:
            cw1, cw2 = int(diffY/2), int(diffY/2)

        x1 = F.pad(x1, (cw1,cw2,ch1,ch2))
        x = torch.cat([x2, x1], dim=1)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
