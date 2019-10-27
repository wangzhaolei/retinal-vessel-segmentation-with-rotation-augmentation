import torch
import torch.nn as nn
import torch.nn.functional as F
class only_conv(nn.Module):
    def __init__(self, base):
        super(only_conv, self).__init__()
        self.vgg = nn.ModuleList(base)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].
        """
        for k in range(14):
            x = self.vgg[k](x)
        return x          
    
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)#padding=6, dilation=6)
    conv7 = nn.Conv2d(128, 2, kernel_size=1)
    layers += [conv6,nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

base = [32, 64, 96, 128, 128]

def build_only_conv(input_channel):
    return only_conv(vgg(base,input_channel))
