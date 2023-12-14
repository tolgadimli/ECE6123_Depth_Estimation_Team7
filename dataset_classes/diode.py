import torch
import torch.utils.data as data
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms
from copy import deepcopy
from tqdm import tqdm
import albumentations as A

#MAX_DEPTH = 304.3991
MAX_DEPTH = 305
RANDOM_DEPTH_COEFF = 10

def get_data_portion(data_dir, portion, t):
    current_dir = os.path.join(data_dir, portion, t)
    current_dir_list = os.listdir(current_dir)
    current_dir_list.sort()

    total_data_portion = []
    for dir in current_dir_list:
        subdir_list = os.listdir(os.path.join(current_dir, dir))
        subdir_list = [d for d in subdir_list if '.txt' not in d]
        subdir_list.sort()
        for sdir in subdir_list:
            images_dir = os.path.join(current_dir, dir, sdir)
            images_name_list = os.listdir(images_dir)
            images_name_list = [ide for ide in images_name_list if '.txt' not in ide]
            images_name_list.sort()

            assert len(images_name_list) % 3 == 0
            m = len(images_name_list) // 3  

            img_dir_list = []
            for i in range(m):
                img_dir_list.append( (os.path.join(images_dir, images_name_list[3*i]),
                                os.path.join(images_dir, images_name_list[3*i+1]),
                                os.path.join(images_dir, images_name_list[3*i+2])  ))
            
            total_data_portion = total_data_portion + img_dir_list

    return total_data_portion

class DIODE_Dataset(data.Dataset):
    def __init__(self, data_dir = 'DIODE', mode = 'train', demo = False, portion = 'train', img_resize = None, depth_resize = None,
                                meanv = (0.417, 0.379, 0.364), stdv =(0.215, 0.214, 0.229), indoor = True, outdoor = True  ):

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
                # A.RandomBrightnessContrast(p=0.3),
                A.Resize(height=int(img_resize[0]*1.2), width=int(img_resize[1]*1.2)),
                A.RandomCrop(height=img_resize[0], width=img_resize[1]),
            ])
        self.target_resize = transforms.Resize(size=depth_resize)

        self.data_portion = []
        if indoor:
            self.data_portion = self.data_portion + get_data_portion(data_dir, portion, 'indoors')
        if outdoor:
            self.data_portion = self.data_portion + get_data_portion(data_dir, portion, 'outdoor')

        if self.data_portion == []:
            raise ValueError('No such data in the specified setting!')


    def __getitem__(self, idx):
        one_tuple = self.data_portion[idx]
        img = np.array(Image.open( one_tuple[0] ) )
        depth = np.load(one_tuple[1] ).squeeze()
        # mask = np.load(one_tuple[2] )

        transformed = self.common_transforms(image = img, mask = depth)
        img, depth = transformed['image'], transformed['mask']

        if self.demo:
            orig_img, orig_depth = deepcopy(img), deepcopy(depth)

        img = transforms.Normalize(self.meanv, self.stdv) (transforms.ToTensor()(img))
        depth =  torch.unsqueeze(torch.as_tensor(depth, dtype=torch.float32), dim = 0)
        depth = self.target_resize(depth).squeeze()
        depth = RANDOM_DEPTH_COEFF * depth / MAX_DEPTH

        if self.demo:
            return img, depth, orig_img, orig_depth
        else:
            return img, depth

    def __len__(self):
        return len(self.data_portion)


def get_mean_and_std_diode(dataset):
    
    print('Calculating mean and std values...')
    imgs = []
    N = len(dataset)
    for idx in tqdm(range(N)):
        img,_ = dataset[idx]
        # print(img)
        # plt.imshow(img)
        # print(img.shape)
        imgs.append(img)
        # if idx == 100:
        #     break

    # print(imgs[0].shape)
    imgs = np.stack(imgs, axis=0)
    print(imgs.shape)
    mean_r = imgs[:,0,:,:].mean()
    mean_g = imgs[:,1,:,:].mean()
    mean_b = imgs[:,2,:,:].mean()

    # calculate std over each channel (r,g,b)
    std_r = imgs[:,0,:,:].std()
    std_g = imgs[:,1,:,:].std()
    std_b = imgs[:,2,:,:].std()

    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)


# train_dataset = DIODE_Dataset(data_dir = '../../datasets/DIODE', mode = 'train', demo = False, portion = 'train', img_resize = (300,400), depth_resize = (150,200))
# m = len(train_dataset)
# max_depth = 0
# for i in range(m):
#     _,d = train_dataset[i]
#     temp_max = torch.max(d)
#     if temp_max > max_depth:
#         max_depth = temp_max
#         print(max_depth)
# print('===============')
# print(max_depth)


# # mean_vector, std_vector = get_mean_and_std_diode(train_dataset)
# # (0.41705924, 0.3793961, 0.36385778), (0.21542302, 0.21402489, 0.22897081)

# m = len(train_dataset)
# max_depth = 0
# for i in range(m):
#     _,d = train_dataset[i]
#     temp_max = torch.max(d)
#     if temp_max > max_depth:
#         max_depth = temp_max
#         print(max_depth)
# print('===============')
# print(max_depth)


# one_tuple = total_data_portion[0]
# img = np.array(Image.open( one_tuple[0] ) )
# depth = np.load(one_tuple[1] )
# mask = np.load(one_tuple[2] )
