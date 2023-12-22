import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt



from options import Options
from dataset_classes import SUNRGBD_Dataset, NYU_Depth_Dataset
from losses import *
from metrics import calc_metrics
from utils import get_model, SID

cmap = plt.cm.viridis
COMMON_SCALAR = 8

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3]

torch.manual_seed(32)
# args = Options().parse()

model_config = 'NYUv2_DORN_uproj_ordinal_2023-12-15_00:00'
mc_ls = model_config.split('_')
model_path = os.path.join('../All_Results', model_config, 'model_params.pt')

if mc_ls[1] == 'FCRN' or mc_ls[1] == 'UpConv':  
    model_name = '_'.join(mc_ls[1:3])
    decoder = mc_ls[3]
else:
    model_name = mc_ls[1]
    decoder = None

dataset_name = mc_ls[0]
device = 'cuda:0'
print(model_name, dataset_name)

# Determining the input image resizes depending on the backbone model
if 'resnet50' in model_name.lower():
    img_resize = (228, 304)
    depth_resize = (128, 160)
elif 'alexnet' in model_name.lower():
    img_resize = (290, 380)
    depth_resize = (128, 160)
elif 'vgg16' in model_name.lower():
    img_resize = (260, 340)
    depth_resize = (128, 160)
elif 'dorn' in model_name.lower():
    img_resize = (257, 353)
    depth_resize = (257, 353)
elif 'adabins' in model_name.lower():
    img_resize = (360, 480)
    depth_resize = (180, 240)
elif 'sarpn' in model_name.lower():
    img_resize = (228, 304)
    depth_resize = (114, 152) 
else:
    raise ValueError('Invalid model type.')



if dataset_name == 'SUN-RGBD':
    train_dataset = SUNRGBD_Dataset('dataset_classes/SUN-RGBD', portion = 'train', transform = None, flip_p = None, img_resize=img_resize, depth_resize=depth_resize, demo = True)
    test_dataset = SUNRGBD_Dataset('dataset_classes/SUN-RGBD', portion = 'test', transform = None, flip_p = None, img_resize=img_resize, depth_resize=depth_resize, demo = True)

elif dataset_name == 'NYUv2':
    train_dataset = NYU_Depth_Dataset(mode = 'eval', demo = True, portion = 'train', img_resize=img_resize, depth_resize=depth_resize)
    test_dataset = NYU_Depth_Dataset(mode = 'eval', demo = True, portion = 'eval', img_resize=img_resize, depth_resize=depth_resize)

# test data 142, 143, 144, 600, 640, 643, 650


model = get_model(model_name, decoder, device)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.to(device)
model.eval()

# idx = 5
for idx in range(143,144):
    sid = SID(device = device)
    inp_image, dm, orig_inp_image, gt_depth_map = test_dataset[idx]
    gt_depth_map = np.asarray(gt_depth_map)
    inp_image = inp_image.unsqueeze(0).to(device)
    with torch.no_grad():
        if model_name == 'DORN':
            pred_labels, pred_softmax = model(inp_image)
            out_depth = sid.labels2depth(pred_labels).squeeze(dim=1)
        else:
            out_depth = model(inp_image)
    pred_dm = torch.squeeze(out_depth).cpu().detach().numpy()
    small_dm = torch.squeeze(dm).cpu().detach().numpy()

    plt.figure(figsize=(10,6), dpi = 120)
    plt.subplot(2,1,1)
    plt.axis('off')
    plt.imshow(small_dm, cmap='jet')
    plt.title('Ground Truth')

    plt.subplot(2,1,2)
    plt.axis('off')
    plt.imshow(pred_dm, cmap='jet')
    plt.title('Prediction')

    plt.figure(figsize=(3,4), dpi = 120)
    plt.axis('off')
    plt.imshow(small_dm, cmap='jet')
    # plt.imsave('nyuv2_eval143_gt.png', small_dm)

    plt.figure(figsize=(3,4), dpi = 120)
    plt.axis('off')
    plt.imshow(pred_dm, cmap='jet')
    # plt.imsave('nyuv2_eval143_%s.png'%model_name, pred_dm )



