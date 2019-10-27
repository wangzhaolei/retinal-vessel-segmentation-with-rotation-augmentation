import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os

def get_crop_shape (inp,orig):
    diffh = orig.size()[2] - inp.size()[2]
    if diffh % 2 != 0:
        if diffh > 0:
            ch1, ch2 = int(diffh/2), int(diffh/2) + 1
        else: ch1, ch2 = int(diffh/2), int(diffh/2) - 1
    else:
        ch1, ch2 = int(diffh/2), int(diffh/2)

    diffw = orig.size()[3] - inp.size()[3]
    if diffw % 2 != 0:
        if diffw > 0:
            cw1, cw2 = int(diffw/2), int(diffw/2) + 1
        else: cw1, cw2 = int(diffw/2), int(diffw/2) - 1
    else:
        cw1, cw2 = int(diffw/2), int(diffw/2)
    return (cw1,cw2,ch1,ch2) 
class DRIU(nn.Module):
    def __init__(self, base):
        super(DRIU, self).__init__()
        self.vgg = nn.ModuleList(base)
        self.convmedia1 = nn.Conv2d(64,16,kernel_size=3, padding=1)
        self.convmedia2 = nn.Conv2d(128,16,kernel_size=3, padding=1)
        self.convmedia3 = nn.Conv2d(256,16,kernel_size=3, padding=1)
        #self.convmedia4 = nn.Conv2d(512,16,kernel_size=3,padding=1)
        self.deconv1 = nn.ConvTranspose2d(16, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(16,16,kernel_size=8, stride=4)
        #self.deconv3 = nn.ConvTranspose2d(16,16,kernel_size=16,stride=8)
        self.convout = nn.Conv2d(48,2,kernel_size=1)
		
    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].
        """
        sources = list()
        # apply vgg up to conv4_3 relu 3,8,15,22
        for k in range(14):
            x = self.vgg[k](x)
            if k in [3,8,13]:
                sources.append(x)

        cm1 = self.convmedia1(sources[0])
        cm2 = self.convmedia2(sources[1])
        cm3 = self.convmedia3(sources[2])
        dc1 = self.deconv1(cm2)
        dc1_crop = F.pad(dc1,get_crop_shape(dc1,cm1))
        dc2 = self.deconv2(cm3)
        dc2_crop = F.pad(dc2,get_crop_shape(dc2,cm1))
    
        cat_feature = torch.cat([cm1,dc1_crop,dc2_crop],dim=1)
        output = self.convout(cat_feature)
        
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
			
	
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

base = [64, 64, 'M', 128, 128, 'M', 256, 256]#[, 256, 'M', 512, 512, 512]

def build_driu(input_channel):
    return DRIU(vgg(base,input_channel))
