from models import FCRN_ResNet50, FCRN_AlexNet, FCRN_VGG16
from models import UpConv_ResNet50, UpConv_AlexNet, UpConv_VGG16
from models import  FCRN_ResNet50v2
from models import DORN
from models import UnetAdaptiveBins
import torch

def get_model(model_name, decoder, device):
    # print(model_name, decoder)
    if model_name.lower() == 'fcrn_resnet50':
        model = FCRN_ResNet50()
    elif model_name.lower() == 'fcrn_alexnet':
        model = FCRN_AlexNet()
    elif model_name.lower() == 'fcrn_vgg16':
        model = FCRN_VGG16()
    elif model_name.lower() == 'upconv_resnet50':
        model = UpConv_ResNet50()
    elif model_name.lower() == 'upconv_alexnet':
        model = UpConv_AlexNet()
    elif model_name.lower() == 'upconv_vgg16':
        model = UpConv_VGG16()
    elif model_name.lower() == 'fcrn_resnet50v2':
        model = FCRN_ResNet50v2(device, decoder)
    elif model_name.lower() == 'dorn':
        model = DORN()
    elif model_name.lower() == 'adabins':
        model = UnetAdaptiveBins.build(n_bins=256)
    else:
        raise ValueError('Invalid model type.')
    
    return model


class SID:
    def __init__(self, device = 'cpu'):
        super(SID, self).__init__()

        alpha = 0.7113
        beta = 9.9955
        K = 80.0

        self.device = device
            
        self.alpha = torch.tensor(alpha).to(device)
        self.beta = torch.tensor(beta).to(device)
        self.K = torch.tensor(K).to(device)
        
    def labels2depth(self, labels):
        depth = self.alpha * (self.beta / self.alpha) ** (labels.float() / self.K)
        return depth.float()

    
    # def depth2labels(self, depth):
    #     labels = self.K * torch.log(depth / self.alpha) / torch.log(self.beta / self.alpha)
    #     return labels.cuda().round().int()

    def depth2labels(self, depth):
        labels = self.K * torch.log(depth / self.alpha) / torch.log(self.beta / self.alpha)
        return labels.to(self.device).round().int()
    
    # def depth2labels2(self, depth):
    #     labels = self.alpha * torch.pow(self.beta / self.alpha, depth/self.K)
    #     return labels.to(self.device) .round().int()