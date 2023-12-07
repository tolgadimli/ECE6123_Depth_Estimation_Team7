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
from utils import get_model

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

model_config = 'NYUv2_UpConv_ResNet50_berhu_2023-11-12_12:18'
mc_ls = model_config.split('_')
model_path = os.path.join('All_Results', model_config, 'model_params.pt')

model_name = '_'.join(model_config.split('_')[1:3])
dataset_name = mc_ls[0]
device = 'cpu'
loss_fnc = 'MSE'
print(model_name, dataset_name)
decoder = None

# Determining the input image resizes depending on the backbone model
if 'resnet50' in model_name.lower():
    img_resize = (228, 304)
elif 'alexnet' in model_name.lower():
    img_resize = (290, 380)
elif 'vgg16' in model_name.lower():
    img_resize = (260, 340)
else:
    raise ValueError('Invalid model type.')
depth_resize = (128, 160)



if dataset_name == 'SUN-RGBD':
    train_dataset = SUNRGBD_Dataset('dataset_classes/SUN-RGBD', portion = 'train', transform = None, flip_p = None, img_resize=img_resize, depth_resize=depth_resize, demo = True)
    test_dataset = SUNRGBD_Dataset('dataset_classes/SUN-RGBD', portion = 'test', transform = None, flip_p = None, img_resize=img_resize, depth_resize=depth_resize, demo = True)

elif dataset_name == 'NYUv2':
    train_dataset = NYU_Depth_Dataset(mode = 'eval', demo = True, portion = 'train', img_resize=img_resize, depth_resize=depth_resize)
    test_dataset = NYU_Depth_Dataset(mode = 'eval', demo = True, portion = 'eval', img_resize=img_resize, depth_resize=depth_resize)




model = get_model(model_name, decoder, device)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.to(device)
model.eval()

# idx = 5

idx = 1
inp_image, dm, orig_inp_image, gt_depth_map = train_dataset[idx]
gt_depth_map = np.asarray(gt_depth_map)
inp_image = inp_image.unsqueeze(0).to(device)
with torch.no_grad():
    pred_dm = model(inp_image)
pred_dm = torch.squeeze(pred_dm).cpu().detach().numpy()
small_dm = torch.squeeze(dm).cpu().detach().numpy()

# plt.figure(figsize=(10,6), dpi = 120)
# plt.subplot(2,1,1)
# plt.axis('off')
# plt.imshow(colored_depthmap(small_dm).astype("uint8"))
# plt.title('Ground Truth')

# plt.subplot(2,1,2)
# plt.axis('off')
# plt.imshow(colored_depthmap(pred_dm).astype("uint8"))
# plt.title('Prediction')



plt.figure(figsize=(10,6), dpi = 120)
plt.subplot(2,1,1)
plt.axis('off')
plt.imshow(small_dm)
plt.title('Ground Truth')

plt.subplot(2,1,2)
plt.axis('off')
plt.imshow(pred_dm)
plt.title('Prediction')
