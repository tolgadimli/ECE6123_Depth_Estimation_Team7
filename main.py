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

from losses import berHuLoss, MaskedL1Loss, OrdinalLoss, SILogLoss, BinsChamferLoss, SARPN_total_loss
from metrics import calc_metrics
from utils import get_model, SID, adjust_gt
from dataset_classes import SUNRGBD_Dataset, NYU_Depth_Dataset, DIODE_Dataset
import sys
sys.path.append('..')

def train(model, device, train_loader, optimizer, scheduler, criterion):

    train_loss = 0
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = torch.squeeze(model(data))
        # print(output.shape, target.shape)

        loss = criterion(output, target)
        # print(loss)

        loss.backward()
        optimizer.step()
        # print(loss.item())
        train_loss = train_loss + loss.item()

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
            output = torch.squeeze(model(data))
            metric_ls = calc_metrics(output, target)
            test_loss += criterion(output, target).item()  # sum up batch loss
            all_metrics = [all_metrics[i] + metric_ls[i] for i in range(8)]

    test_loss /= len(test_loader.dataset)
    all_metrics = [all_metrics[i] / len(test_loader) for i in range(8)]

    return test_loss, all_metrics


def train_dorn(model, device, train_loader, optimizer, scheduler, criterion, sid):

    train_loss = 0
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred_labels, pred_softmax = model(data)
        target_labels = sid.depth2labels(target).unsqueeze(dim=1)
        loss = criterion(pred_softmax, target_labels)

        loss.backward()
        optimizer.step()

        train_loss = train_loss + loss.item()

    scheduler.step()
    train_loss = train_loss / len(train_loader.dataset)
    
    
    return train_loss

def test_dorn(model, device, test_loader, criterion, sid):

    model.eval()
    test_loss = 0
    all_metrics = [0] * 8
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target_labels = sid.depth2labels(target).unsqueeze(dim=1)
            
            pred_labels, pred_softmax = model(data)
            out_depth = sid.labels2depth(pred_labels).squeeze(dim=1)

            #print(target.shape, out_depth.shape)
            metric_ls = calc_metrics(out_depth, target)
            test_loss += criterion(pred_softmax, target_labels).item()  # sum up batch loss
            all_metrics = [all_metrics[i] + metric_ls[i] for i in range(8)]

    test_loss /= len(test_loader.dataset)
    all_metrics = [all_metrics[i] / len(test_loader) for i in range(8)]

    return test_loss, all_metrics




def train_adabins(model, device, train_loader, optimizer, scheduler, criterion1, criterion2):

    train_loss = 0
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        bin_edges, pred = model(data)
        pred = pred.squeeze()

        # print(pred.shape, target.shape, bin_edges.shape)
        loss1 = criterion1(pred, target)
        loss2 = criterion2(bin_edges, target)
        # print(loss.item())
        loss = loss1 + 0.1 * loss2
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        
        optimizer.step()
        train_loss = train_loss + loss.item()
        scheduler.step()
    train_loss = train_loss / len(train_loader.dataset)
    
    return train_loss

def test_adabins(model, device, test_loader, criterion1, criterion2):

    model.eval()
    test_loss = 0
    all_metrics = [0] * 8
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            bin_edges, pred = model(data)
            pred = pred.squeeze()

            # print(target.shape, pred.shape)
            loss1 = criterion1(pred, target)
            loss2 = criterion2(bin_edges, target)
            loss = loss1 + 0.1 * loss2

            metric_ls = calc_metrics(pred, target)
            test_loss += loss.item()  # sum up batch loss
            all_metrics = [all_metrics[i] + metric_ls[i] for i in range(8)]

    test_loss /= len(test_loader.dataset)
    all_metrics = [all_metrics[i] / len(test_loader) for i in range(8)]

    return test_loss, all_metrics



def train_sarpn(model, device, train_loader, optimizer, scheduler, criterion):

    train_loss = 0
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        target = target.unsqueeze(dim=1)
        # print(target.shape)
        image = torch.autograd.Variable(data)
        target = torch.autograd.Variable(target)	

        pred_depth = model(image)
        # print([pred_depth[i].shape for i in range(len(pred_depth))])
        gt_depth = adjust_gt(target, pred_depth)

        loss = criterion(gt_depth, pred_depth)
        print(loss)

        loss.backward()
        optimizer.step()

        train_loss = train_loss + loss.item()
        break

    scheduler.step()
    train_loss = train_loss / len(train_loader.dataset)
    
    
    return train_loss

def test_sarpn(model, device, test_loader, criterion):

    model.eval()
    test_loss = 0
    all_metrics = [0] * 8
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.unsqueeze(dim=1)

            pred_depth = model(data)
            gt_depth = adjust_gt(target, pred_depth)

            loss = criterion(gt_depth, pred_depth)

            print(pred_depth[4].squeeze(dim=1).shape, gt_depth[4].squeeze(dim=1).shape)
            metric_ls = calc_metrics(pred_depth[4].squeeze(dim=1), gt_depth[4].squeeze(dim=1))
            test_loss += loss.item()  # sum up batch loss
            all_metrics = [all_metrics[i] + metric_ls[i] for i in range(8)]

            break

    test_loss /= len(test_loader.dataset)
    all_metrics = [all_metrics[i] / len(test_loader) for i in range(8)]

    return test_loss, all_metrics


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == '__main__':

    torch.manual_seed(32)
    args = Options().parse()
    model_name = args.model_name
    dataset_name = args.dataset_name
    device = args.device
    loss_fnc = args.loss_fnc
    decoder = args.decoder

    print(args.bs)

    load_model_dir = args.load_model_dir
    save_model = True

    print(model_name, decoder, loss_fnc)

    # MAIN_EXPERIMENT_DIR = 'All_Results'
    # if MAIN_EXPERIMENT_DIR not in os.listdir('..'):
    #     os.mkdir(os.path.join('..', MAIN_EXPERIMENT_DIR))

    # date_time = ':'.join(str(datetime.datetime.now()).split(':')[0:2])
    # date_time = date_time.replace(' ', '_')
    # expr_config = '%s_%s_%s_%s_%s'%(dataset_name, model_name, decoder, loss_fnc, date_time)
    # expr_dir = os.path.join('..', MAIN_EXPERIMENT_DIR, expr_config)
    # print(expr_dir)

    # if expr_config not in os.listdir(os.path.join('..', MAIN_EXPERIMENT_DIR)):
    #     os.mkdir(expr_dir)


    MAIN_EXPERIMENT_DIR = 'All_Results'
    if MAIN_EXPERIMENT_DIR not in os.listdir():
        os.mkdir(MAIN_EXPERIMENT_DIR)

    date_time = ':'.join(str(datetime.datetime.now()).split(':')[0:2])
    date_time = date_time.replace(' ', '_')
    expr_config = '%s_%s_%s_%s_%s'%(dataset_name, model_name, decoder, loss_fnc, date_time)
    expr_dir = os.path.join(MAIN_EXPERIMENT_DIR, expr_config)
    print(expr_dir)

    if expr_config not in os.listdir( MAIN_EXPERIMENT_DIR):
        os.mkdir(expr_dir)        

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
        img_resize = (300, 400)
        depth_resize = (150, 200) 
    else:
        raise ValueError('Invalid model type.')
    

    model = get_model(model_name, decoder, device)
    if load_model_dir != '':
        param_dir = os.path.join(MAIN_EXPERIMENT_DIR,load_model_dir, 'model_params.pt')
        model.load_state_dict(torch.load(param_dir, map_location='cpu'))
        print('Model parameters from %s are loaded.'%param_dir)

    model.to(device)

    if loss_fnc.lower() == 'mse':
        criterion = nn.MSELoss(reduction = 'mean')
    elif loss_fnc.lower() == 'berhu':
        print('using berHuLoss') 
        criterion = berHuLoss()
    elif 'l1' in loss_fnc.lower():
        print('using MaskedL1Loss')
        criterion = MaskedL1Loss()
    elif 'ordinal' in loss_fnc.lower():
        criterion = OrdinalLoss(device)
    elif 'silog' in loss_fnc.lower():
        criterion_silog = SILogLoss()
        criterion_chamfer = BinsChamferLoss()
    elif 'sarpn' in loss_fnc.lower():
        criterion = SARPN_total_loss



    sid = SID(device = device)

    if dataset_name == 'SUN-RGBD':
        epoch = 80
        lr_drops = [40,60]
        train_dataset = SUNRGBD_Dataset('datasets/SUN-RGBD', mode = 'train', demo = False, portion = 'train', img_resize=img_resize, depth_resize=depth_resize)
        test_dataset = SUNRGBD_Dataset('datasets/SUN-RGBD', mode = 'eval', demo = False, portion = 'eval', img_resize=img_resize, depth_resize=depth_resize)

    elif dataset_name == 'NYUv2':
        epoch = 25
        lr_drops = [15,20]
        train_dataset = NYU_Depth_Dataset(mode = 'train', demo = False, portion = 'train', img_resize=img_resize, depth_resize=depth_resize)
        test_dataset = NYU_Depth_Dataset(mode = 'eval', demo = False, portion = 'eval', img_resize=img_resize, depth_resize=depth_resize)

    elif dataset_name == 'DIODE':
        epoch = 80
        lr_drops = [40,60]
        train_dataset = DIODE_Dataset(data_dir = 'datasets/DIODE', mode = 'train', demo = False, portion = 'train',img_resize=img_resize, depth_resize=depth_resize )
        test_dataset = DIODE_Dataset(data_dir = 'datasets/DIODE', mode = 'eval', demo = False, portion = 'val', img_resize=img_resize, depth_resize=depth_resize )

    train_loader = data.DataLoader(train_dataset, batch_size=args.bs, shuffle = True)
    test_loader  = data.DataLoader(test_dataset, batch_size=args.bs, shuffle = False)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if 'dorn' in model_name.lower():
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr}, {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        optimizer = torch.optim.SGD(train_params, lr=args.lr, weight_decay=args.wd, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_drops, gamma=0.1)

    elif 'adabins' in model_name.lower():
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr}, {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, epochs=epoch, steps_per_epoch=len(train_loader), cycle_momentum=True,
                                                                    base_momentum=0.85, max_momentum=0.95, div_factor=25, final_div_factor=100)    

    elif 'sarpn' in model_name.lower():
        if dataset_name == 'SUN-RGBD':
            lr_drops = [20,40,60]
        elif dataset_name == 'NYUv2':
            lr_drops = [7,14,20]
        elif dataset_name == 'DIODE':
            lr_drops = [20,40,60]
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_drops, gamma=0.1)  

    else:                          
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_drops, gamma=0.1)



    train_errs, test_errs = [], []

    generic_headers = ['epoch', 'train_loss', 'test_loss' ]
    new_metric_list = ['rel_mae', 'rel_mse', 'rmse_linear', 'rmse_log', 'silog', 'a1', 'a2', 'a3', 'lr']
    csv_cols = generic_headers + new_metric_list
    
    df_content= []
    for e in tqdm(range(1, epoch+1)):
        if 'ordinal' in loss_fnc.lower():
            train_loss = train_dorn(model, device, train_loader, optimizer, scheduler, criterion, sid)
            test_loss, all_metrics = test_dorn(model, device, test_loader, criterion, sid)
        elif 'adabins' in model_name.lower():
            train_loss = train_adabins(model, device, train_loader, optimizer, scheduler, criterion_silog, criterion_chamfer)
            test_loss, all_metrics = test_adabins(model, device, test_loader, criterion_silog, criterion_chamfer)

        elif 'sarpn' in model_name.lower():
            train_loss = train_sarpn(model, device, train_loader, optimizer, scheduler, criterion)
            test_loss, all_metrics = test_sarpn(model, device, test_loader, criterion)

        else:
            train_loss = train(model, device, train_loader, optimizer, scheduler, criterion)
            test_loss, all_metrics = test(model, device, test_loader, criterion)
        current_lr = get_lr(optimizer)
        df_content.append([e, train_loss, test_loss] + all_metrics + [current_lr])

        df = pd.DataFrame(df_content, columns = csv_cols)
        df.to_csv(os.path.join(expr_dir, 'results.csv'))

        if save_model:
            save_path = os.path.join(expr_dir, 'model_params.pt')
            torch.save(model.state_dict(), save_path) 