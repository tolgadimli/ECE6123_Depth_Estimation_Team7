import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

# from blocks import *


import torch
import torch.nn as nn
import torchvision

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FastUpProjectionBlock(nn.Module):
    '''
    Fast Up Projection Block as proposed in "Deeper Depth Prediction with
    Fully Convolutional Residual Networks" by Laina I. et al.
    '''
    def __init__(self, in_channels, out_channels):
        super(FastUpProjectionBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # First Branch
        self.convA11 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(3, 3))
        self.convA12 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(2, 3))
        self.convA13 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(3, 2))
        self.convA14 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(2, 2))

        self.reluA2 = nn.ReLU(inplace=True)
        self.batchnormA2 = nn.BatchNorm2d(out_channels)

        self.convA3 = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=(3, 3),
                                padding=(1, 1))
        self.batchnormA3 = nn.BatchNorm2d(out_channels)

        # Second Branch
        self.convB11 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(3, 3))
        self.convB12 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(2, 3))
        self.convB13 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(3, 2))
        self.convB14 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(2, 2))
        self.batchnormB2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inp):
        '''
        inp: Tensor of shape (batch_size, in_channels, H, W)
        out: Tensor of shape (batch_size, out_channels, H', W')
        '''

        inpA11 = F.pad(self.convA11(inp), pad=(1, 1, 1, 1))
        inpA12 = F.pad(self.convA12(inp), pad=(1, 1, 1, 0))
        inpA13 = F.pad(self.convA13(inp), pad=(1, 0, 1, 1))
        inpA14 = F.pad(self.convA14(inp), pad=(1, 0, 1, 0))
        outA1 = torch.cat((inpA11, inpA12), dim=2)
        outA2 = torch.cat((inpA13, inpA14), dim=2)
        outA = torch.cat((outA1, outA2), dim=3)
        outA = self.reluA2(self.batchnormA2(outA))
        outA = self.batchnormA3(self.convA3(outA))

        inpB11 = F.pad(self.convB11(inp), pad=(1, 1, 1, 1))
        inpB12 = F.pad(self.convB12(inp), pad=(1, 1, 1, 0))
        inpB13 = F.pad(self.convB13(inp), pad=(1, 0, 1, 1))
        inpB14 = F.pad(self.convB14(inp), pad=(1, 0, 1, 0))

        outB1 = torch.cat((inpB11, inpB12), dim=2)
        outB2 = torch.cat((inpB13, inpB14), dim=2)

        outB = torch.cat((outB1, outB2), dim=3)
        outB = self.batchnormB2(outB)

        out = self.relu(outA + outB)

        return out


class FCRN_ResNet50(nn.Module):
    '''
    FCRN: Fully Convolutional Residual Network based on a ResNet-50 architecture.

    Implementation follows architecture of FCRN in "Deeper Depth Prediction with
    Fully Convolutional Residual Networks" by Laina I. et al.
    '''
    def __init__(self):
        super(FCRN_ResNet50, self).__init__()

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
        # Fast Up Projection Blocks
        decoder['UpProj1'] = FastUpProjectionBlock(in_channels=512 * 2, out_channels=512)
        decoder['UpProj2'] = FastUpProjectionBlock(in_channels=256 * 2, out_channels=256)
        decoder['UpProj3'] = FastUpProjectionBlock(in_channels=128 * 2, out_channels=128)
        decoder['UpProj4'] = FastUpProjectionBlock(in_channels=64 * 2, out_channels=64)

        # Out Convolution Layer
        conv3 = nn.Conv2d(in_channels=64,
                          out_channels=1,
                          kernel_size=(3, 3),
                          padding=(1, 1))

        relu = nn.ReLU()
        decoder['Conv3'] = nn.Sequential(conv3, relu)

        self.decoder = nn.Sequential(decoder)

    def forward(self, inp):
        '''
        inp: Tensor of shape (batch_size, in_channels, H, W) 
        out: Tensor of shape (batch_size, 1, H', W')
        '''
        out = self.encoder(inp)
        # print(out.shape)
        return self.decoder(out)


class FCRN_VGG16(nn.Module): # 
    '''
    FCRN: Fully Convolutional Residual Network based on a VGG16 architecture.
    Requires input size of B x 3 x 260 x 340 to produce B x 1 x 128 x 160
    '''
    def __init__(self):
        super(FCRN_VGG16, self).__init__()

        decoder = OrderedDict()
        # Encoder
        # In Convolution Layer
        vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        vgg16_modules = list(vgg16.children())[0]

        # Convolution Layer
        conv2 = nn.Conv2d(in_channels=512,
                          out_channels=256,
                          kernel_size=(1, 1))
        batchnorm2 = nn.BatchNorm2d(256)

        vgg_modules_added = list(vgg16_modules) + [conv2, batchnorm2]
        self.encoder = nn.Sequential(*vgg_modules_added)

        # Decoder
        # Fast Up Projection Blocks
        decoder['UpProj1'] = FastUpProjectionBlock(in_channels=256, out_channels=128)
        decoder['UpProj2'] = FastUpProjectionBlock(in_channels=128, out_channels=64)
        decoder['UpProj3'] = FastUpProjectionBlock(in_channels=64, out_channels=32)
        decoder['UpProj4'] = FastUpProjectionBlock(in_channels=32, out_channels=16)

        # Out Convolution Layer
        conv3 = nn.Conv2d(in_channels=16,
                          out_channels=1,
                          kernel_size=(3, 3),
                          padding=(1, 1))

        relu = nn.ReLU()
        decoder['Conv3'] = nn.Sequential(conv3, relu)

        self.decoder = nn.Sequential(decoder)

    def forward(self, inp):
        '''
        inp: Tensor of shape (batch_size, in_channels, H, W) 
        out: Tensor of shape (batch_size, 1, H', W')
        '''
        out = self.encoder(inp)
        # print(out.shape)
        return self.decoder(out)


class FCRN_AlexNet(nn.Module):
    '''
    FCRN: Fully Convolutional Residual Network based on a AlexNet architecture.
    Requires input size of B x 3 x 290 x 380 to produce B x 1 x 128 x 160
    '''
    def __init__(self):
        super(FCRN_AlexNet, self).__init__()

        decoder = OrderedDict()
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
        # Fast Up Projection Blocks
        decoder['UpProj1'] = FastUpProjectionBlock(in_channels=128, out_channels=64)
        decoder['UpProj2'] = FastUpProjectionBlock(in_channels=64, out_channels=32)
        decoder['UpProj3'] = FastUpProjectionBlock(in_channels=32, out_channels=16)
        decoder['UpProj4'] = FastUpProjectionBlock(in_channels=16, out_channels=8)

        # Out Convolution Layer
        conv3 = nn.Conv2d(in_channels=8,
                          out_channels=1,
                          kernel_size=(3, 3),
                          padding=(1, 1))

        relu = nn.ReLU()
        decoder['Conv3'] = nn.Sequential(conv3, relu)

        self.decoder = nn.Sequential(decoder)

    def forward(self, inp):
        '''
        inp: Tensor of shape (batch_size, in_channels, H, W) 
        out: Tensor of shape (batch_size, 1, H', W')
        '''
        out = self.encoder(inp)
        # print(out.shape)
        return self.decoder(out)



# if __name__ == '__main__':
#     x = torch.randn(10, 3, 290, 380)
#     fcrn = FCRN_AlexNet()
#     x = fcrn(x)
#     print(x.shape)

    # vgg16 = torchvision.models.vgg16_bn(pretrained=True)
    # vgg16_modules = list(vgg16.children())[0]
    # vgg16_feat = nn.Sequential(*vgg16_modules)

    # alexnet = torchvision.models.alexnet(pretrained=True)
    # alexnet_modules = list(alexnet.children())[0]
    # alexnet_features = nn.Sequential(*alexnet_modules)
    # x = torch.randn(10, 3, 228, 304)