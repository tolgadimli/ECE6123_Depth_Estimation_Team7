import torch
import torch.utils.data as data
from torchvision import transforms
from datasets import load_dataset

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy

from PIL import Image
import albumentations as A
import logging

MAX_VAL = 10
COMMON_SCALAR = 2.5

class NYU_Depth_Dataset(data.Dataset):
    def __init__(self, mode = 'train', demo = False, portion = 'train', img_resize = None, depth_resize = None,
                                meanv = (0.466 , 0.391, 0.373), stdv =(0.273, 0.275 , 0.288) ):
        super().__init__()

        self.portion = portion
        self.img_resize, self.depth_resize = img_resize, depth_resize
        self.demo = demo
        self.meanv, self.stdv = meanv, stdv

        if mode == 'train':
            self.common_transforms = A.Compose([
                A.Rotate(5),
                A.HorizontalFlip(p=0.5),
                # A.RandomBrightnessContrast(p=0.3),
                A.Resize(height=int(img_resize[0]*1.2), width=int(img_resize[1]*1.2)),
                A.RandomCrop(height=img_resize[0], width=img_resize[1]),
            ])
        elif mode == 'eval':
            self.common_transforms = A.Compose([
                A.Resize(height=int(img_resize[0]*1.2), width=int(img_resize[1]*1.2)),
                A.CenterCrop(height=img_resize[0], width=img_resize[1]),
            ])

        self.target_resize = transforms.Resize(size=depth_resize)

        if portion == 'train':
            self.dataset = load_dataset("sayakpaul/nyu_depth_v2", split="train")
        elif portion == 'eval' or portion == 'test':
            self.dataset = load_dataset("sayakpaul/nyu_depth_v2", split="validation")

        self.length = len(self.dataset)

    def __getitem__(self, idx):
        
        img, depth = np.array(self.dataset[idx]['image']), np.array(self.dataset[idx]['depth_map'])

        #img, depth = np.array(self.dataset[idx]['image'], dtype=np.float32) / 255.0, np.array(self.dataset[idx]['depth_map'], dtype=np.float32) / 1000.0
        #print(img)
        transformed = self.common_transforms(image = img, mask = depth)
        img, depth = transformed['image'], transformed['mask']

        if self.demo:
            orig_img, orig_depth = deepcopy(img), deepcopy(depth)

        img = transforms.Normalize(self.meanv, self.stdv) (transforms.ToTensor()(img))
        depth = torch.unsqueeze(torch.as_tensor(depth, dtype=torch.float32), dim = 0)
        depth = self.target_resize(depth).squeeze()

        if self.demo:
            return img, depth, orig_img, orig_depth
        else:
            return img, depth

    def __len__(self):
        return self.length

def get_mean_and_std(dataset):
    
    print('Calculating mean and std values...')
    mean_ls, std_ls = [], []
    imgs = []
    N = len(dataset)
    for idx in tqdm(range(N)):
        _, _, img, _ = dataset[idx]
        # print(img)
        # plt.imshow(img)
        # print(img.shape)
        imgs.append(img / 255.0)
        if (idx+1) % 5000 == 0 or idx+1 == N:

            imgs = np.stack(imgs, axis=0)
            # print(imgs.shape)
            mean_r = imgs[:,:,:,0].mean()
            mean_g = imgs[:,:,:,1].mean()
            mean_b = imgs[:,:,:,2].mean()

            # calculate std over each channel (r,g,b)
            std_r = imgs[:,:,:,0].std()
            std_g = imgs[:,:,:,1].std()
            std_b = imgs[:,:,:,2].std()
            mean_ls.append((mean_r, mean_g, mean_b))
            std_ls.append((std_r, std_g, std_b))
            imgs = []

    return mean_ls, std_ls


# if __name__ == '__main__':

#     img_resize = (384, 480)
#     depth_resize = (192, 240)
#     dataset = NYU_Depth_Dataset(mode = 'train', demo = False, portion = 'train', img_resize=img_resize, depth_resize=depth_resize)
#     ult_max = 0
#     for i in tqdm(range(len(dataset))):
#         img, dp = dataset[i]
#         # b = transforms.ToTensor()(dp)
#         cur_max = np.max(np.asarray(dp))
#         # print(cur_max)
#         ult_max = max(ult_max, cur_max)

#         if i % 1000 == 0:
#             print(ult_max)
#         break

#     print(ult_max)

#     meanv = (0.4656348 , 0.39099635, 0.37311448)
#     stdv = (0.27271997, 0.2749485 , 0.2881334)

#     # idx = 41624
#     # train_dataset = load_dataset("sayakpaul/nyu_depth_v2", split="train")
#     # img, depth = np.array(train_dataset[idx]['image']), np.array(train_dataset[idx]['depth_map'])

#     # resizes = (228, 304)
#     # transform = A.Compose([ A.Resize(width=int(resizes[1]*1.2), height=int(resizes[0]*1.2)),
#     #             A.CenterCrop(width=resizes[1], height=resizes[0])])  
             
#     img_resize = (228, 304)
#     depth_resize = (128, 160)
#     meanv = (0.466 , 0.391, 0.373)
#     stdv = (0.273, 0.275 , 0.288)
#     dataset = NYU_Depth_Dataset(mode = 'eval', demo = False, portion = 'eval', img_resize=img_resize, depth_resize=depth_resize)
    
    # a,b =dataset[0]
    # plt.figure()
    # plt.imshow(b)
    # a_fixed = a * torch.tensor((0.25, 0.25, 0.25)).unsqueeze(1).unsqueeze(2) + torch.tensor((0.5, 0.5, 0.5)).unsqueeze(1).unsqueeze(2)
    # plt.figure()
    # plt.imshow(np.transpose(a_fixed.numpy(), axes=(1,2,0)))


    # logging.basicConfig(filename="nyuv2_stats2.log",
    #                 format='%(asctime)s %(message)s',
    #                 filemode='w')
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
 
    # dataset = NYU_Depth_Dataset(mode = 'eval', demo = True, portion = 'train', img_resize = (228, 304), depth_resize = (128, 160))
    # meanv, stdv = get_mean_and_std(dataset)
    # logger.info("(228, 304)")
    # logger.info("Mean: " + str(meanv))
    # logger.info("Std: " + str(stdv))
    # # for (228, 304): 

    # means = [(0.48564205805406185, 0.4053763404803231, 0.3602511270662459), (0.4579809911194397, 0.37391067676932144, 0.36211691385810846), (0.459740643059856, 0.3797988246722991, 0.36629444608522194), (0.4975229544384669, 0.4238411547331306, 0.4234221300219072), (0.4545881401404046, 0.379594858282186, 0.3803889197241686), (0.4605530465595745, 0.38683674639707905, 0.3663166080060823), (0.4694992685305875, 0.3831989066997999, 0.33602565646895605), (0.4457408014117458, 0.36847588883683735, 0.36327698750067616), (0.4850506852900497, 0.41920166425642186, 0.39707281721298826), (0.4400294259934074, 0.38972846010636814, 0.37597918185002116)]
    # stds = [(0.2523331080626479, 0.2433804555640725, 0.24711624550272296), (0.26128350541051976, 0.25866945382885215, 0.2719677600615312), (0.26901100857717697, 0.2693783324270261, 0.2870824715819529), (0.26958070222930425, 0.28231615306535485, 0.29572167241378605), (0.2790645752533772, 0.2802010394047263, 0.29094654985035995), (0.2760367511257935, 0.2783376946937608, 0.2910411509768691), (0.2797558802611981, 0.2778443275556493, 0.28945997628652936), (0.2879217620904387, 0.29461008911668235, 0.3113755734727858), (0.28298357258829065, 0.2897756925132331, 0.30558270643597085), (0.2692287888540405, 0.27497174356234644, 0.29103986573105556)]
    # mean_act, std_act = 0, 0 
    # for meanv, stdv in zip(means, stds):
    #     mean_act = mean_act + np.array(meanv)
    #     std_act = std_act + np.array(stdv)
    # mean_act = mean_act/len(means)
    # std_act = std_act/len(stds)

    #meanv, stdv = get_mean_and_std(train_dataset, transform)

    # transformed = transform(image = img, mask = depth)
    # timg, target = transformed['image'], transformed['mask']

    # target = target.astype('float32')
    # img = transforms.Normalize(self.mean_vector, self.std_vector) (transforms.ToTensor()(img))
    # target = torch.as_tensor(target, dtype=torch.float32)

    # img_resize = transforms.Compose([transforms.Resize((240, 320)), transforms.CenterCrop((228, 304))])
    # # depth_resize = transforms.Resize(size = (128, 160))
    # depth_resize = transforms.Compose([transforms.Resize((134, 168)), transforms.CenterCrop((128, 160))])
    # dataset = NYU_Depth_Dataset(portion = 'test', transform = None, flip_p=0, img_resize=img_resize, depth_resize=depth_resize, demo=True)
    # img, depth, orig_img, orig_depth = dataset[0]


# plt.imshow(np.transpose(np.asarray(img),(1,2,0)))