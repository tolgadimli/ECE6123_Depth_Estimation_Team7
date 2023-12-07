import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

# from blocks import *


import torch
import torch.nn as nn
import torchvision


class UpConv_ResNet50(nn.Module):
    '''
    Requires 228 x 304 to produce output of size 128 x 160
    '''
    def __init__(self):
        super(UpConv_ResNet50, self).__init__()

        decoder = OrderedDict()
        # Encoder
        # In Convolution Layer
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet_modules = list(resnet.children())[:-2]

        # Convolution Layer
        conv2 = nn.Conv2d(in_channels=512 * 4,
                          out_channels=512 * 2,
                          kernel_size=(1, 1))
        batchnorm2 = nn.BatchNorm2d(512 * 2)

        resnet_modules_added = resnet_modules + [conv2, batchnorm2]
        self.encoder = nn.Sequential(*resnet_modules_added)

        # Decoder
        relu = nn.ReLU()
        decoder_modules = []
        for i in range(0, 4):
            ups = nn.Upsample(scale_factor = 2, mode = 'bilinear')
            conv = nn.Conv2d(in_channels=2**(10-i), out_channels=2**(9-i), kernel_size=(5, 5), padding = 2)
            btc = nn.BatchNorm2d(2**(9-i))
            decoder_modules = decoder_modules + [ups, conv, relu, btc]

        conv_final = nn.Conv2d(in_channels=64,
                          out_channels=1,
                          kernel_size=(3, 3),
                          padding=(1, 1))
        
        decoder_modules = decoder_modules + [conv_final, relu]
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, inp):
        '''
        inp: Tensor of shape (batch_size, in_channels, H, W) 
        out: Tensor of shape (batch_size, 1, H', W')
        '''
        out = self.encoder(inp)
        # print(out.shape)
        return self.decoder(out)



class UpConv_VGG16(nn.Module):
    '''
    Requires 228 x 304 to produce output of size 128 x 160
    '''
    def __init__(self):
        super(UpConv_VGG16, self).__init__()

        # Encoder
        # In Convolution Layer
        vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        vgg16_modules = list(vgg16.children())[0]

        # Convolution Layer
        conv2 = nn.Conv2d(in_channels=512,
                          out_channels=256,
                          kernel_size=(1, 1))
        batchnorm2 = nn.BatchNorm2d(256)

        vgg16_modules_added = list(vgg16_modules) + [conv2, batchnorm2]
        self.encoder = nn.Sequential(*vgg16_modules_added)

        # Decoder
        relu = nn.ReLU()
        decoder_modules = []
        for i in range(0, 4):
            ups = nn.Upsample(scale_factor = 2, mode = 'bilinear')
            conv = nn.Conv2d(in_channels=2**(8-i), out_channels=2**(7-i), kernel_size=(5, 5), padding = 2)
            btc = nn.BatchNorm2d(2**(7-i))
            decoder_modules = decoder_modules + [ups, conv, relu, btc]

        conv_final = nn.Conv2d(in_channels=16,
                          out_channels=1,
                          kernel_size=(3, 3),
                          padding=(1, 1))
        
        decoder_modules = decoder_modules + [conv_final, relu]
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, inp):
        '''
        inp: Tensor of shape (batch_size, in_channels, H, W) 
        out: Tensor of shape (batch_size, 1, H', W')
        '''
        out = self.encoder(inp)
        # print(out.shape)
        return self.decoder(out)


class UpConv_AlexNet(nn.Module):
    '''
    Requires 228 x 304 to produce output of size 128 x 160
    '''
    def __init__(self):
        super(UpConv_AlexNet, self).__init__()

        # Encoder
        # In Convolution Layer
        alexnet = torchvision.models.alexnet(pretrained=True)
        alexnet_modules = list(alexnet.children())[0]

        # Convolution Layer
        conv2 = nn.Conv2d(in_channels=256,
                          out_channels=128,
                          kernel_size=(1, 1))
        batchnorm2 = nn.BatchNorm2d(128)

        alexnet_modules_added = list(alexnet_modules) + [conv2, batchnorm2]
        self.encoder = nn.Sequential(*alexnet_modules_added)

        # Decoder
        relu = nn.ReLU()
        decoder_modules = []
        for i in range(0, 4):
            ups = nn.Upsample(scale_factor = 2, mode = 'bilinear')
            conv = nn.Conv2d(in_channels=2**(7-i), out_channels=2**(6-i), kernel_size=(5, 5), padding = 2)
            btc = nn.BatchNorm2d(2**(6-i))
            decoder_modules = decoder_modules + [ups, conv, relu, btc]

        conv_final = nn.Conv2d(in_channels=8,
                          out_channels=1,
                          kernel_size=(3, 3),
                          padding=(1, 1))
        
        decoder_modules = decoder_modules + [conv_final, relu]
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, inp):
        '''
        inp: Tensor of shape (batch_size, in_channels, H, W) 
        out: Tensor of shape (batch_size, 1, H', W')
        '''
        out = self.encoder(inp)
        # print(out.shape)
        return self.decoder(out)


# if __name__ == '__main__':
#     in_channels = 3
#     x = torch.randn(10, in_channels, 290, 380)
#     fcrn = UpConv_AlexNet(in_channels=in_channels)
#     x = fcrn(x)
#     print(x.shape)
