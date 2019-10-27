# full assembly of the sub-parts to form the complete net
from unet_parts import *
#from unet_parts_orig import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np
class UNet_level4_our(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_level4_our, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 256)
        self.up1 = up(512,256)
        self.up2 = up(384,128) #256+128,128
        self.up3 = up(192,64) #128+64,64
        self.up4 = up(96,32) #64+32,32
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)

        self.up1 = up(192, 64)
        self.up2 = up(96, 32)
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        #np.save('./xinput.npy',x.data.cpu().numpy())
        x1 = self.inc(x)
        #np.save('./x1.npy',x1.data.cpu().numpy())
        x2 = self.down1(x1)
        #np.save('./x2.npy',x2.data.cpu().numpy())
        x3 = self.down2(x2)
        #np.save('./x3.npy',x3.data.cpu().numpy())
        x4 = self.up1(x3, x2)
        #np.save('./x4.npy',x4.data.cpu().numpy())
        x5 = self.up2(x4, x1)
        #np.save('./x5.npy',x5.data.cpu().numpy())
        x6 = self.outc(x5)
        #np.save('./x6.npy',x6.data.cpu().numpy())
        #assert (x.size()[1]==x6.size()[1])
        return x6

class UNet_side_loss(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_side_loss, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)

        self.up1 = up(192, 64)
        self.up2 = up(96, 32)
        self.outc = outconv(32, n_classes)
        self.outc1=outconv(32,n_classes)
        self.outc2=outconv(64,n_classes)
        self.outc3=outconv(128,n_classes)
        self.outc4=outconv(64,n_classes)

    def forward(self, x):
        
        x1 = self.inc(x)
        x1out=self.outc1(x1)
        
        x2 = self.down1(x1)
        x2out=self.outc2(F.upsample(x2,scale_factor=2))
        
        x3 = self.down2(x2)
        x3out=self.outc3(F.upsample(x3,scale_factor=4))
        
        x4 = self.up1(x3, x2)
        x4out=self.outc4(F.upsample(x4,scale_factor=2))
        
        x5 = self.up2(x4, x1)
        
        x6 = self.outc(x5)
        
        return x6,x1out,x2out,x3out,x4out

class UNet_cat(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_cat, self).__init__()
        #self.inc = inconv(n_channels, 32)
        self.inc1 = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.inc2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)

        self.up1 = up(192, 64)
        self.up2 = up(96, 32)
        self.outc = outconv(352, n_classes)

    def forward(self, x):
        #np.save('./xinput.npy',x.data.cpu().numpy())
        x1_1 = self.inc1(x)
        x1=self.inc2(x1_1)
        #np.save('./x1.npy',x1.data.cpu().numpy())
        x2 = self.down1(x1)
        #np.save('./x2.npy',x2.data.cpu().numpy())
        x3 = self.down2(x2)
        #np.save('./x3.npy',x3.data.cpu().numpy())
        x4 = self.up1(x3, x2)
        #np.save('./x4.npy',x4.data.cpu().numpy())
        x5 = self.up2(x4, x1)
        #np.save('./x5.npy',x5.data.cpu().numpy())
        x_fuse=torch.cat((x1_1,x1,
            F.upsample(x2,scale_factor=2,mode='bilinear',align_corners=False),
            F.upsample(x3,scale_factor=4,mode='bilinear',align_corners=False),
            F.upsample(x4,scale_factor=2,mode='bilinear',align_corners=False),
            x5),1)
        x6 = self.outc(x_fuse) #32+32+64+128+64+32
        #np.save('./x6.npy',x6.data.cpu().numpy())
        #assert (x.size()[1]==x6.size()[1])
        return x6
class UNet_cat_with_side(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_cat_with_side, self).__init__()
        #self.inc = inconv(n_channels, 32)
        self.inc1 = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.inc2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)

        self.up1 = up(192, 64)
        self.up2 = up(96, 32)
        self.outc = outconv(352, n_classes)
        self.outc1=outconv(32,n_classes)
        self.outc2=outconv(64,n_classes)
        self.outc3=outconv(128,n_classes)
        self.outc4=outconv(64,n_classes)

    def forward(self, x):
        
        x1_1 = self.inc1(x)
        x1=self.inc2(x1_1)
        x1out=self.outc1(x1)

        x2 = self.down1(x1)
        x2up=F.upsample(x2,scale_factor=2)
        x2out=self.outc2(x2up)

        x3 = self.down2(x2)
        x3up=F.upsample(x3,scale_factor=4)
        x3out=self.outc3(x3up)

        x4 = self.up1(x3, x2)
        x4up=F.upsample(x4,scale_factor=2)
        x4out=self.outc4(x4up)

        x5 = self.up2(x4, x1)
        
        x_fuse=torch.cat((x1_1,x1,x2up,x3up,x4up,x5),1)
        x6 = self.outc(x_fuse) #32+32+64+128+64+32
        
        return x6,x1out,x2out,x3out,x4out
class UNet_cat_with_side_bilinear(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_cat_with_side_bilinear, self).__init__()
        #self.inc = inconv(n_channels, 32)
        self.inc1 = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.inc2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)

        self.up1 = up(192, 64)
        self.up2 = up(96, 32)
        self.outc = outconv(352, n_classes)
        self.outc1=outconv(32,n_classes)
        self.outc2=outconv(64,n_classes)
        self.outc3=outconv(128,n_classes)
        self.outc4=outconv(64,n_classes)

    def forward(self, x):
        
        x1_1 = self.inc1(x)
        x1=self.inc2(x1_1)
        x1out=self.outc1(x1)

        x2 = self.down1(x1)
        x2up=F.upsample(x2,scale_factor=2,mode='bilinear',align_corners=False)
        x2out=self.outc2(x2up)

        x3 = self.down2(x2)
        x3up=F.upsample(x3,scale_factor=4,mode='bilinear',align_corners=False)
        x3out=self.outc3(x3up)

        x4 = self.up1(x3, x2)
        x4up=F.upsample(x4,scale_factor=2,mode='bilinear',align_corners=False)
        x4out=self.outc4(x4up)

        x5 = self.up2(x4, x1)
        
        x_fuse=torch.cat((x1_1,x1,x2up,x3up,x4up,x5),1)
        x6 = self.outc(x_fuse) #32+32+64+128+64+32
        
        return x6,x1out,x2out,x3out,x4out

class UNet_cat_simple(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_cat_simple, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)

        self.up1 = up(192, 64)
        self.up2 = up(96, 32)
        self.outc = outconv(320, n_classes)

    def forward(self, x):
        #np.save('./xinput.npy',x.data.cpu().numpy())
        x1= self.inc(x)
        #np.save('./x1.npy',x1.data.cpu().numpy())
        x2 = self.down1(x1)
        #np.save('./x2.npy',x2.data.cpu().numpy())
        x3 = self.down2(x2)
        #np.save('./x3.npy',x3.data.cpu().numpy())
        x4 = self.up1(x3, x2)
        #np.save('./x4.npy',x4.data.cpu().numpy())
        x5 = self.up2(x4, x1)
        #np.save('./x5.npy',x5.data.cpu().numpy())
        x_fuse=torch.cat((x1,
            F.upsample(x2,scale_factor=2,mode='bilinear',align_corners=False),
            F.upsample(x3,scale_factor=4,mode='bilinear',align_corners=False),
            F.upsample(x4,scale_factor=2,mode='bilinear',align_corners=False),
            x5),1)
        x6 = self.outc(x_fuse) #32+64+128+64+32
        #np.save('./x6.npy',x6.data.cpu().numpy())
        #assert (x.size()[1]==x6.size()[1])
        return x6
class UNet_collaborative_small(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_collaborative_small, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)

        self.up1_1 = up(192, 64)
        self.up1_2=up(192,64)
        self.up2_1 = up(96, 32)
        self.up2_2=up(96,32)
        self.up2_3 = up(96, 32)
        self.up2_4=up(96,32)
        self.outc1 = outconv(32, n_classes)
        self.outc2 = outconv(32, n_classes)
        self.outc3 = outconv(32, n_classes)
        self.outc4 = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4_1 = self.up1_1(x3, x2)
        x4_2=self.up1_2(x3,x2)
        x5_1 = self.up2_1(x4_1, x1)
        x5_2 = self.up2_2(x4_1, x1)
        x5_3 = self.up2_3(x4_2, x1)
        x5_4 = self.up2_4(x4_2, x1)
        x6_1 = self.outc1(x5_1)
        x6_2 = self.outc2(x5_2)
        x6_3 = self.outc3(x5_3)
        x6_4 = self.outc4(x5_4)
        return x6_1,x6_2,x6_3,x6_4
class UNet_collaborative(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_collaborative, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down11 = down(32, 64)
        self.down12 = down(32, 64)

        self.down21 = down(64, 128)
        self.down22 = down(64, 128)
        self.down23 = down(64, 128)
        self.down24 = down(64, 128)

        self.up11 = up(192, 64)
        self.up12 = up(192, 64)
        self.up13 = up(192, 64)
        self.up14 = up(192, 64)
        self.up15 = up(192, 64)
        self.up16 = up(192, 64)
        self.up17 = up(192, 64)
        self.up18 = up(192, 64)

        self.up21 = up(96, 32)
        self.up22 = up(96, 32)
        self.up23 = up(96, 32)
        self.up24 = up(96, 32)
        self.up25 = up(96, 32)
        self.up26 = up(96, 32)
        self.up27 = up(96, 32)
        self.up28 = up(96, 32)
        self.up29 = up(96, 32)
        self.up210 = up(96, 32)
        self.up211 = up(96, 32)
        self.up212 = up(96, 32)
        self.up213 = up(96, 32)
        self.up214 = up(96, 32)
        self.up215 = up(96, 32)
        self.up216 = up(96, 32)

        self.outc1 = outconv(32, n_classes)
        self.outc2 = outconv(32, n_classes)
        self.outc3 = outconv(32, n_classes)
        self.outc4 = outconv(32, n_classes)
        self.outc5 = outconv(32, n_classes)
        self.outc6 = outconv(32, n_classes)
        self.outc7 = outconv(32, n_classes)
        self.outc8 = outconv(32, n_classes)
        self.outc9 = outconv(32, n_classes)
        self.outc10 = outconv(32, n_classes)
        self.outc11 = outconv(32, n_classes)
        self.outc12 = outconv(32, n_classes)
        self.outc13 = outconv(32, n_classes)
        self.outc14 = outconv(32, n_classes)
        self.outc15 = outconv(32, n_classes)
        self.outc16 = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2_1 = self.down11(x1)
        x2_2=self.down12(x1)

        x3_1 = self.down21(x2_1)
        x3_2 = self.down22(x2_1)
        x3_3 = self.down23(x2_2)
        x3_4 = self.down24(x2_2)

        x4_1 = self.up11(x3_1, x2_1)
        x4_2 = self.up12(x3_1, x2_1)
        x4_3 = self.up13(x3_2, x2_1)
        x4_4 = self.up14(x3_2, x2_1)
        x4_5 = self.up15(x3_3, x2_2)
        x4_6 = self.up16(x3_3, x2_2)
        x4_7 = self.up17(x3_4, x2_2)
        x4_8 = self.up18(x3_4, x2_2)

        x5_1 = self.up21(x4_1, x1)
        x5_2 = self.up22(x4_1, x1)
        x5_3 = self.up23(x4_2, x1)
        x5_4 = self.up24(x4_2, x1)
        x5_5 = self.up25(x4_3, x1)
        x5_6 = self.up26(x4_3, x1)
        x5_7 = self.up27(x4_4, x1)
        x5_8 = self.up28(x4_4, x1)
        x5_9 = self.up29(x4_5, x1)
        x5_10 = self.up210(x4_5, x1)
        x5_11 = self.up211(x4_6, x1)
        x5_12 = self.up212(x4_6, x1)
        x5_13 = self.up213(x4_7, x1)
        x5_14 = self.up214(x4_7, x1)
        x5_15 = self.up215(x4_8, x1)
        x5_16 = self.up216(x4_8, x1)

        x6_1 = self.outc1(x5_1)
        x6_2 = self.outc2(x5_2)
        x6_3 = self.outc3(x5_3)
        x6_4 = self.outc4(x5_4)
        x6_5 = self.outc5(x5_5)
        x6_6 = self.outc6(x5_6)
        x6_7 = self.outc7(x5_7)
        x6_8 = self.outc8(x5_8)
        x6_9 = self.outc9(x5_9)
        x6_10 = self.outc10(x5_10)
        x6_11 = self.outc11(x5_11)
        x6_12 = self.outc12(x5_12)
        x6_13 = self.outc13(x5_13)
        x6_14 = self.outc14(x5_14)
        x6_15 = self.outc15(x5_15)
        x6_16 = self.outc16(x5_16)
        return x6_1,x6_2,x6_3,x6_4,x6_5,x6_6,x6_7,x6_8,x6_9,x6_10,x6_11,x6_12,x6_13,x6_14,x6_15,x6_16



class UNet_big_input(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_big_input, self).__init__()
        self.inc = inconv_orig(n_channels, 64)
        self.down1 = down_orig(64, 128)
        self.down2 = down_orig(128, 256)

        self.up1 = up_orig(256, 128,bilinear=False) #256/2+128
        self.up2 = up_orig(128, 64,bilinear=False) #128/2+64
        self.outc = outconv_orig(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return x
class UNet_largekernel(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_largekernel, self).__init__()
        self.inc = inconv_1(n_channels, 256)
        self.down1 = down(256, 256)
        self.down2 = down(256, 128)

        self.up1 = up(384, 128)
        self.up2 = up(384, 128)
        self.outc = outconv(128, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return x
class UNet_multichannel(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_multichannel, self).__init__()
        self.inc = inconv(n_channels, 256)
        self.down1 = down(256, 256)
        self.down2 = down(256, 128)

        self.up1 = up(384, 128)
        self.up2 = up(384, 128)
        self.outc = outconv(128, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return x

class UNet_down_convstride_2(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_down_convstride_2, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down_convstride_2(32, 64)
        self.down2 = down_convstride_2(64, 128)
        
        self.up1 = up(192, 64)
        self.up2 = up(96, 32)
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        #print('doubleconv1 size:',x1.size())
        x2 = self.down1(x1)
        #print('maxpool1 size:',x2.size())
        x3 = self.down2(x2)
        #print('maxpool2 size',x3.size())
        x = self.up1(x3, x2)
        #print('up1',x.size())
        x = self.up2(x, x1)
        #print('up2',x.size())
        x = self.outc(x)
        #print('out',x.size())
        return x

class UNet_up_deconv(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_up_deconv, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down_convstride_2(32, 64)
        self.down2 = down_convstride_2(64, 128)
        
        self.up1 = up(192, 64,bilinear=False)
        self.up2 = up(96, 32,bilinear=False)
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        #print('doubleconv1 size:',x1.size())
        x2 = self.down1(x1)
        #print('maxpool1 size:',x2.size())
        x3 = self.down2(x2)
        #print('maxpool2 size',x3.size())
        x = self.up1(x3, x2)
        #print('up1',x.size())
        x = self.up2(x, x1)
        #print('up2',x.size())
        x = self.outc(x)
        #print('out',x.size())
        return x
class UNet_channel_end(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_channel_end, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        
        self.up1 = up(192, 64)
        self.up2 = up(96, 32)
        self.outc = outconv(33, n_classes)

    def forward(self, x,channel):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = torch.cat([x,channel], dim=1)
        x = self.outc(x)
        return x


class UNet_again(nn.Module):
    def __init__(self, n_channels,n_channels_1, n_classes):
        super(UNet_again, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.inc_1 = inconv(n_channels_1,32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        
        self.up1 = up(192, 64)
        self.up2 = up(96, 32)
        self.outc = outconv(32, n_classes)

    def forward(self, x_in):
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x_up1 = self.up1(x3, x2)
        x_up2 = self.up2(x_up1, x1)
        x_out1 = self.outc(x_up2)

        x1_again = self.inc_1(torch.cat([x_in,torch.unsqueeze(x_out1[:,1,:,:],1)],dim=1))
        x2_again = self.down1(x1_again)
        x3_again = self.down2(x2_again)
        x_up1_again = self.up1(x3_again, x2_again)
        x_up2_again = self.up2(x_up1_again, x1_again)
        x_out2 = self.outc(x_up2_again)
        return x_out1,x_out2
class UNet_2branch(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_2branch, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        
        self.up1 = up(192, 64)
        self.up2 = up(128, 32)
        self.outc = outconv(32, n_classes)

        self.up3 = up(96,32)

    def forward(self, x):
        x1_1 = self.inc(x)

        x2_1 = self.down1(x1_1)
        x1_2=self.up3(x2_1,x1_1)

        x3_1 = self.down2(x2_1)
        x2_2 = self.up1(x3_1, x2_1)

        x_media=torch.cat([x1_1, x1_2], dim=1)

        x1_3 = self.up2(x2_2, x_media)

        x_out1=self.outc(x1_2)
        x_out2 = self.outc(x1_3)
        
        return x_out1,x_out2

class UNet_plus(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_plus, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128,256)
        
        self.up1 = up(192, 64)
        self.up2 = up(128, 32)
        self.outc = outconv(32, n_classes)

        self.up3 = up(96,32)

        self.up4 = up(384,128)
        self.up5 = up(256,64)
        self.up6 = up(160,32)

    def forward(self, x):
        x1_1 = self.inc(x)

        x2_1 = self.down1(x1_1)
        x1_2=self.up3(x2_1,x1_1) #out is 32


        x3_1 = self.down2(x2_1)
        x2_2 = self.up1(x3_1, x2_1)  #out is 64
        x1_3 = self.up2(x2_2, torch.cat([x1_1, x1_2], dim=1)) #out is 32



        x4_1=self.down3(x3_1)
        x3_2=self.up4(x4_1,x3_1)#384 to 128
        x2_3=self.up5(x3_2,torch.cat([x2_1,x2_2], dim=1)) #128+64+64 to 64
        x1_4=self.up6(x2_3,torch.cat([x1_1, x1_2,x1_3], dim=1)) #64+32*3 to 32



        x_out1=self.outc(x1_2)
        x_out2 = self.outc(x1_3)
        x_out3=self.outc(x1_4)
        
        return x_out1,x_out2,x_out3
 
 
class UNet_plus_4(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_plus_4, self).__init__()
        self.inc = inconv(n_channels,64)
        self.down1 = down(64,128)
        self.down2 = down(128,256)
        self.down3 = down(256,512)
        self.down4 = down(512,512)
        
        self.up1 = up(192,64)
        self.up2 = up(384, 128)
        self.up3 = up(256, 64)
        self.up4 = up(768,256)
        self.up5 = up(512,128)
        self.up6 = up(320,64)
        self.up7 = up(1024,512)
        self.up8 = up(1024,256)
        self.up9 = up(640,128)
        self.up10 = up(384,64)

        self.outc = outconv(64, n_classes)
        self.outc1 = outconv(320, n_classes)
    def forward(self, x):
        x1_1 = self.inc(x)

        x2_1 = self.down1(x1_1)
        x1_2=self.up1(x2_1,x1_1) #(128+64,64)


        x3_1 = self.down2(x2_1)
        x2_2 = self.up2(x3_1, x2_1)  #(256+128, 128)
        x1_3 = self.up3(x2_2, torch.cat([x1_1,x1_2], dim=1)) #(128+64*2, 64)



        x4_1=self.down3(x3_1)
        x3_2=self.up4(x4_1,x3_1)#(256+512,256)
        x2_3=self.up5(x3_2,torch.cat([x2_1,x2_2], dim=1)) #(256+128*2,128)
        x1_4=self.up6(x2_3,torch.cat([x1_1,x1_2,x1_3], dim=1)) #(128+64*3,64)


        x5_1=self.down4(x4_1)
        x4_2=self.up7(x5_1,x4_1)#(512+512,512)
        x3_3=self.up8(x4_2,torch.cat([x3_1,x3_2],dim=1))# (512+256*2,256)
        x2_4=self.up9(x3_3,torch.cat([x2_1,x2_2,x2_3], dim=1)) # (256+128*3,128)
        x1_5=self.up10(x2_4,torch.cat([x1_1,x1_2,x1_3,x1_4], dim=1)) #(128+64*4,64)


        x_out1=self.outc(x1_2)
        x_out2 = self.outc(x1_3)
        x_out3=self.outc(x1_4)
        x_out4=self.outc(x1_5)
        #x_fuse=x_out4+x_out1+x_out2+x_out3
        #x_fuse=self.outc1(torch.cat([x_out1,x_out2,x_out3,x_out4],dim=1))
        x_fuse=self.outc1(torch.cat([x1_1,x1_2,x1_3,x1_4,x1_5],dim=1))
        return x_out1,x_out2,x_out3,x_out4,x_fuse    
class UNet_plus_4_orig(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_plus_4_orig, self).__init__()
        self.inc = inconv(n_channels,64)
        self.down1 = down(64,128)
        self.down2 = down(128,256)
        self.down3 = down(256,512)
        self.down4 = down(512,512)
        
        self.up1 = up(192,64)
        self.up2 = up(384, 128)
        self.up3 = up(256, 64)
        self.up4 = up(768,256)
        self.up5 = up(512,128)
        self.up6 = up(320,64)
        self.up7 = up(1024,512)
        self.up8 = up(1024,256)
        self.up9 = up(640,128)
        self.up10 = up(384,64)

        self.outc = outconv(64, n_classes)
    def forward(self, x):
        x1_1 = self.inc(x)

        x2_1 = self.down1(x1_1)
        x1_2=self.up1(x2_1,x1_1) #(128+64,64)


        x3_1 = self.down2(x2_1)
        x2_2 = self.up2(x3_1, x2_1)  #(256+128, 128)
        x1_3 = self.up3(x2_2, torch.cat([x1_1,x1_2], dim=1)) #(128+64*2, 64)



        x4_1=self.down3(x3_1)
        x3_2=self.up4(x4_1,x3_1)#(256+512,256)
        x2_3=self.up5(x3_2,torch.cat([x2_1,x2_2], dim=1)) #(256+128*2,128)
        x1_4=self.up6(x2_3,torch.cat([x1_1,x1_2,x1_3], dim=1)) #(128+64*3,64)


        x5_1=self.down4(x4_1)
        x4_2=self.up7(x5_1,x4_1)#(512+512,512)
        x3_3=self.up8(x4_2,torch.cat([x3_1,x3_2],dim=1))# (512+256*2,256)
        x2_4=self.up9(x3_3,torch.cat([x2_1,x2_2,x2_3], dim=1)) # (256+128*3,128)
        x1_5=self.up10(x2_4,torch.cat([x1_1,x1_2,x1_3,x1_4], dim=1)) #(128+64*4,64)


        x_out1=self.outc(x1_2)
        x_out2 = self.outc(x1_3)
        x_out3=self.outc(x1_4)
        x_out4=self.outc(x1_5)
        
        
        return x_out1,x_out2,x_out3,x_out4
class UNet_level4(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_level4, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 512)
        self.up2 = up(768, 256)
        self.up3 = up(384, 128)
        self.up4 = up(192, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
class UNet_pad_replicate(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_pad_replicate, self).__init__()
        self.inc = inconv_rep(n_channels, 32)
        self.down1 = down_rep(32, 64)
        self.down2 = down_rep(64, 128)
        
        self.up1 = up_rep(192, 64)
        self.up2 = up_rep(96, 32)
        self.outc = outconv_rep(32, n_classes)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return x
