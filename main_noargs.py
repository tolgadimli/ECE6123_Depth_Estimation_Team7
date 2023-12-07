import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import datetime
import pandas as pd

from options import Options
from dataset_classes import SUNRGBD_Dataset, NYU_Depth_Dataset
from losses import *
from metrics import calc_metrics
from utils import get_model


def train(model, device, train_loader, optimizer, scheduler, criterion):

    train_loss = 0
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(output[0])

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        # print(loss.item())
        train_loss = train_loss + loss.item()
        break

    scheduler.step()
    train_loss = train_loss / len(train_loader.dataset)
    
    return train_loss


def test(model, device, test_loader, criterion):

    model.eval()
    test_loss = 0
    all_metrics = [0] * 8
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            metric_ls = calc_metrics(output, target)
            test_loss += criterion(output, target).item()  # sum up batch loss
            all_metrics = [all_metrics[i] + metric_ls[i] for i in range(8)]
            break

    test_loss /= len(test_loader.dataset)
    all_metrics = [all_metrics[i] / len(test_loader) for i in range(8)]

    return test_loss, all_metrics



if __name__ == '__main__':

    torch.manual_seed(32)
    # args = Options().parse()
    model_name = 'UpConv_VGG16'
    dataset_name = 'NYUv2'
    device = 'cuda:0'
    loss_fnc = 'berhu'
    save_model = True

    MAIN_EXPERIMENT_DIR = 'All_Results'
    if MAIN_EXPERIMENT_DIR not in os.listdir():
        os.mkdir(MAIN_EXPERIMENT_DIR)

    date_time = ':'.join(str(datetime.datetime.now()).split(':')[0:2])
    date_time = date_time.replace(' ', '_')
    expr_config = '%s_%s_%s_%s'%(dataset_name, model_name, loss_fnc, date_time)
    expr_dir = os.path.join(MAIN_EXPERIMENT_DIR, expr_config)

    if expr_config not in os.listdir(MAIN_EXPERIMENT_DIR):
        os.mkdir(expr_dir)

    # Determining the input image resizes depending on the backbone model
    if 'resnet50' in model_name.lower():
        img_resize = transforms.Compose([transforms.Resize((240, 320)), transforms.CenterCrop((228, 304))])
    elif 'alexnet' in model_name.lower():
        img_resize = transforms.Compose([transforms.Resize((306, 400)), transforms.CenterCrop((290, 380))])
    elif 'vgg16' in model_name.lower():
        img_resize = transforms.Compose([transforms.Resize((272, 356)), transforms.CenterCrop((260, 340))])
    else:
        raise ValueError('Invalid model type.')
    depth_resize = transforms.Compose([transforms.Resize((134, 168)), transforms.CenterCrop((128, 160))])

    model = get_model(model_name)
    model.to(device)

    if loss_fnc == 'MSE':
        criterion = nn.MSELoss(reduction = 'sum')
    elif loss_fnc.lower() == 'berhu':
        criterion = BerHuLoss()

    common_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, hue=0.1),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1))])

    if dataset_name == 'SUN-RGBD':
        epoch = 100
        lr_drops = [50,75]
        train_dataset = SUNRGBD_Dataset('dataset_classes/SUN-RGBD', portion = 'train', transform = common_transforms, flip_p = 0.5, img_resize=img_resize, depth_resize=depth_resize)
        test_dataset = SUNRGBD_Dataset('dataset_classes/SUN-RGBD', portion = 'test', transform = None, flip_p = None, img_resize=img_resize, depth_resize=depth_resize)

    elif dataset_name == 'NYUv2':
        epoch = 20
        lr_drops = [8,16]
        train_dataset = NYU_Depth_Dataset(portion = 'train', transform = common_transforms, flip_p = 0.5, img_resize=img_resize, depth_resize=depth_resize)
        test_dataset = NYU_Depth_Dataset(portion = 'test', transform = None, flip_p = None, img_resize=img_resize, depth_resize=depth_resize)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_drops, gamma=0.1)

    train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle = True)
    test_loader  = data.DataLoader(test_dataset, batch_size=16, shuffle = False)

    train_errs, test_errs = [], []

    generic_headers = ['epoch', 'train_loss', 'test_loss' ]
    new_metric_list = ['rel_mae', 'rel_mse', 'rmse_linear', 'rmse_log', 'silog', 'a1', 'a2', 'a3']
    csv_cols = generic_headers + new_metric_list

    df_content= []
    for e in tqdm(range(1, epoch+1)):
        train_loss = train(model, device, train_loader, optimizer, scheduler, criterion)
        test_loss, all_metrics = test(model, device, test_loader, criterion)
        df_content.append([e, train_loss, test_loss] + all_metrics)

        df = pd.DataFrame(df_content, columns = csv_cols)
        df.to_csv(os.path.join(expr_dir, 'results.csv'))

    if save_model:
        save_path = os.path.join(expr_dir, 'model_params.pt')
        torch.save(model.state_dict(), save_path)