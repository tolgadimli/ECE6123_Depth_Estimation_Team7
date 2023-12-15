import torch
import torch.utils.data as data
import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms
from copy import deepcopy

from tqdm import tqdm
import albumentations as A

from PIL import Image
import logging

MAX_DEPTH = 65536/10
RANDOM_DEPTH_COEFF = 2.5

# applying horizontal flip together
def horizontal_flip(img, depth, p):
    r = torch.rand(1).item()
    if r < p:
        return transforms.RandomHorizontalFlip(p=1)(img), transforms.RandomHorizontalFlip(p=1)(depth)
    else:
        return img, depth


class SUNRGBD_Dataset(data.Dataset):
    def __init__(self, data_dir = 'SUN-RGBD', mode = 'train', demo = False, portion = 'train', img_resize = None, depth_resize = None,
                                meanv = (0.5, 0.5, 0.5), stdv =(0.25, 0.25, 0.25)  ):
        super().__init__()


        self.data_dir = data_dir
        self.img_resize, self.depth_resize = img_resize, depth_resize
        self.meanv, self.stdv = meanv, stdv
        self.mode = mode
        self.demo = demo

        if mode == 'eval':
            self.common_transforms = A.Compose([
                A.Resize(height=int(img_resize[0]*1.2), width=int(img_resize[1]*1.2)),
                A.CenterCrop(height=img_resize[0], width=img_resize[1]),
            ])

        elif mode == 'train':
            self.common_transforms = A.Compose([
                A.Rotate(5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Resize(height=int(img_resize[0]*1.2), width=int(img_resize[1]*1.2)),
                A.RandomCrop(height=img_resize[0], width=img_resize[1]),
            ])

            

        self.target_resize = transforms.Resize(size=depth_resize)

        if portion == 'train':
            self.dirs = pd.read_csv( os.path.join(data_dir, 'alltrain.csv'))['directories']
        elif portion == 'eval' or portion == 'test':
            self.dirs = pd.read_csv( os.path.join(data_dir, 'alltest.csv'))['directories']

        self.length = len(self.dirs)

    def __getitem__(self, idx):
        sample_dir = os.path.join(self.data_dir, self.dirs[idx])
        img_root_dir, depth_root_dir = os.path.join(sample_dir, 'image'), os.path.join(sample_dir, 'depth_bfx')
        img_dir = os.listdir(os.path.join(sample_dir, 'image'))[0] 
        depth_dir = os.listdir(os.path.join(sample_dir, 'depth_bfx'))[0]

        img = np.array(Image.open((os.path.join(img_root_dir, img_dir))) )
        depth = np.array(Image.open( ( os.path.join(depth_root_dir, depth_dir) ) ) )

        transformed = self.common_transforms(image = img, mask = depth)
        img, depth = transformed['image'], transformed['mask']

        if self.demo:
            orig_img, orig_depth = deepcopy(img), deepcopy(depth)

        img = transforms.Normalize(self.meanv, self.stdv) (transforms.ToTensor()(img))
        depth =  torch.unsqueeze(torch.as_tensor(depth, dtype=torch.float32), dim = 0)
        depth = self.target_resize(depth).squeeze()
        depth = depth / MAX_DEPTH

        if self.demo:
            return img, depth, orig_img, orig_depth
        else:
            return img, depth

    def __len__(self):
        return self.length


def get_mean_and_std_sunrgbd(dataset):
    
    print('Calculating mean and std values...')
    imgs = []
    N = len(dataset)
    for idx in tqdm(range(N)):
        _, _, img, _ = dataset[idx]
        # print(img)
        # plt.imshow(img)
        # print(img.shape)
        imgs.append(img / 255.0)
        # if idx == 1000:
        #     break

    # print(imgs[0].shape)
    imgs = np.stack(imgs, axis=0)
    # print(imgs.shape)
    mean_r = imgs[:,:,:,0].mean()
    mean_g = imgs[:,:,:,1].mean()
    mean_b = imgs[:,:,:,2].mean()

    # calculate std over each channel (r,g,b)
    std_r = imgs[:,:,:,0].std()
    std_g = imgs[:,:,:,1].std()
    std_b = imgs[:,:,:,2].std()

    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)

if __name__ == '__main__':
    
    dataset = SUNRGBD_Dataset(mode = 'eval', demo = False, portion = 'train', img_resize = (228, 304), depth_resize = (128, 160))
    img, depth = dataset[0]

    # overall_max = 0
    # for i in range(len(dataset)):
    #     img, depth = dataset[i]
    #     cur_max = torch.max(depth)
    #     if cur_max > overall_max:
    #         overall_max = cur_max
    #         print(overall_max)

    # print(overall_max)

    # logging.basicConfig(filename="sunrgbd_stats.log",
    #                 format='%(asctime)s %(message)s',
    #                 filemode='w')
    # logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)
 
    # dataset = SUNRGBD_Dataset(mode = 'eval', demo = True, portion = 'train', img_resize = (228, 304), depth_resize = (128, 160))
    # meanv, stdv = get_mean_and_std_sunrgbd(dataset)
    # logger.info("(228, 304)")
    # logger.info("Mean: " + str(meanv))
    # logger.info("Std: " + str(stdv))
    # # for (228, 304): meanv = (0.492, 0.454, 0.427), stdv = (0.278, 0.286, 0.290)

    # dataset = SUNRGBD_Dataset(mode = 'eval', demo = True, portion = 'train', img_resize = (290, 380), depth_resize = (128, 160))
    # meanv, stdv = get_mean_and_std_sunrgbd(dataset)
    # logger.info("(290, 380)")
    # logger.info("Mean: " + str(meanv))
    # logger.info("Std: " + str(stdv))

    # dataset = SUNRGBD_Dataset(mode = 'eval', demo = True, portion = 'train', img_resize = (260, 340), depth_resize = (128, 160))
    # meanv, stdv = get_mean_and_std_sunrgbd(dataset)
    # logger.info("(260, 340)")
    # logger.info("Mean: " + str(meanv))
    # logger.info("Std: " + str(stdv))