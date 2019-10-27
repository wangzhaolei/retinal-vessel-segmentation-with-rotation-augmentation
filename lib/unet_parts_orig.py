# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv_orig(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv_orig, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv_orig(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv_orig, self).__init__()
        self.conv = double_conv_orig(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down_orig(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_orig, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            double_conv_orig(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up_orig(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up_orig, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch,int(in_ch/2),kernel_size=2, stride=2)

        self.conv = double_conv_orig(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        #print('deconvolution:',x1.size())
        diffX = x1.size()[2] - x2.size()[2]  #diffX is negtive
        if diffX % 2 != 0:
            ch1, ch2 = int(diffX/2), int(diffX/2) - 1
        else:
            ch1, ch2 = int(diffX/2), int(diffX/2)

        diffY = x1.size()[3] - x2.size()[3]  #diffY is negtive
        if diffY % 2 != 0:
            cw1, cw2 = int(diffY/2), int(diffY/2) - 1
        else:
            cw1, cw2 = int(diffY/2), int(diffY/2)

        x2 = F.pad(x2, (cw1,cw2,ch1,ch2))
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class outconv_orig(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv_orig, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch,kernel_size=1,padding=0)

    def forward(self, x):
        x = self.conv(x)
        return x
