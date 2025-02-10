import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path
import json

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision import datasets
import timm.optim.optim_factory as optim_factory
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_mae_PromptGIP_CNN_Head
import dataset.lowlevel_PromtGIP_dataloader as lowlevel_prompt_dataloader
from evaluation_prompt import *
from sklearn.metrics import mean_squared_error

zscore_normalizations_sevir = {
    'vil':{'scale':47.54,'shift':33.44},
    'ir069':{'scale':1174.68,'shift':-3683.58},
    'ir107':{'scale':2562.43,'shift':-1552.80},
    'lght':{'scale':0.60517,'shift':0.02990},
    'vis':{'scale':2259.96,'shift':-1347.91}
}

max_min_satellite = {
    'pct37':{'max':350.559, 'min':138.643},
    'pct89':{'max':320.947, 'min':54.002}
}
def get_args():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--model', default='mae_vit_small_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--output_dir', default='../output_dir/')
    parser.add_argument('--data_path_HQ')
    parser.add_argument('--data_path_LQ')
    parser.add_argument('--dataset_type')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--tta_option', default=0, type=int)
    parser.add_argument('--ckpt', help='resume from checkpoint')
    
    parser.add_argument('--prompt_id', default=-1, type=int)
    parser.add_argument('--degradation_type', default='GaussianNoise', type=str)

    parser.set_defaults(autoregressive=False)
    return parser

@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)

@torch.jit.script
def latitude_weighting_factor_torch(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat))/s

def weighted_acc_torch(pred: torch.Tensor, target: torch.Tensor, data_mask=None) -> torch.Tensor:
    # import pdb
    # pdb.set_trace()
    num_lat = 128
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    
    if data_mask is not None:
        numerator  = torch.sum(weight * pred*data_mask* target, dim=(-1,-2))
        denominator = torch.sqrt(torch.sum(weight * pred*data_mask* pred, dim=(-1,-2)) * torch.sum(weight * target* data_mask* target, dim=(-1,-2)))
        return numerator/denominator
    else:
        # import pdb
        # pdb.set_trace()
        result = torch.sum(weight * pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum(weight * pred **2, dim=(-1,-2)) * torch.sum(weight * target ** 2, dim=(-1,-2)))
        return result

def weighted_rmse_torch(pred: torch.Tensor, target: torch.Tensor, data_mask=None) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = 128
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))

    if data_mask is not None:
        result = torch.sqrt(torch.sum(weight * ((pred - target)*data_mask)**2, dim=(-1,-2))/(data_mask.sum()+1e-10))
        return result
    else:
        result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
        return result

def compute_val(path='vil'):

    folder_path = Path('/mnt/petrelfs/zhaoxiangyu1/code/weather_prompt_new/experiments/ood_test/vil_inter_15min/val_test/Epoch1')
    predict_files = []
    target_files = []
    for file in folder_path.glob('predict*'):
        predict_files.append(str(file))
    for file in folder_path.glob('target*'):
        target_files.append(str(file))
    predict_files.sort()
    target_files.sort()
    print(len(predict_files))
    assert len(predict_files)==len(target_files)
    #! 每隔一个epoch跑一个 validation 步骤
    rmse = 0.0
    idx = 0
    if path == 'vil':
        thresholds = [16, 74, 133, 160, 181, 219]
    elif path == 'ir':
        thresholds = [-6000,-4000,0,2000]
    elif path == 'ir069':
        thresholds = [-4000, -5000, -6000, -7000]
    elif path == 'vis':
        thresholds = [2000,3200,4400,5600,6800]
    elif path == 'no2':
        thresholds = [1,5,10,15]
    elif path == 'rainnet':
        thresholds = [5,10,15,20]
    elif path == 'sat':
        thresholds = [140,160,180,200]

    total_pod = {thr: 0.0 for thr in thresholds}
    total_far = {thr: 0.0 for thr in thresholds}
    total_csi = {thr: 0.0 for thr in thresholds}
    cal_num = 0
    for idx in range(len(predict_files)):
        val_pred = (torch.from_numpy(np.load(predict_files[idx])))
        val_label = (torch.from_numpy(np.load(target_files[idx])))
        # if val_label.mean() <= 100:
        #     continue
        # else:
        cal_num = cal_num+1
        if path=='ir':
            pred = (val_pred-zscore_normalizations_sevir['ir107']['shift'])/zscore_normalizations_sevir['ir107']['scale']
            target = (val_label-zscore_normalizations_sevir['ir107']['shift'])/zscore_normalizations_sevir['ir107']['scale']
        elif path=='vil':
            pred = (val_pred-zscore_normalizations_sevir['vil']['shift'])/zscore_normalizations_sevir['vil']['scale']
            target = (val_label-zscore_normalizations_sevir['vil']['shift'])/zscore_normalizations_sevir['vil']['scale']
        elif path=='ir069':
            pred = (val_pred-zscore_normalizations_sevir['ir069']['shift'])/zscore_normalizations_sevir['ir069']['scale']
            target = (val_label-zscore_normalizations_sevir['ir069']['shift'])/zscore_normalizations_sevir['ir069']['scale']
        elif path == 'vis':
            pred = (val_pred-zscore_normalizations_sevir['vis']['shift'])/zscore_normalizations_sevir['vis']['scale']
            target = (val_label-zscore_normalizations_sevir['vis']['shift'])/zscore_normalizations_sevir['vis']['scale']
        elif path == 'rainnet':
            pred = val_pred/140
            target = val_label/140
        elif path == 'sat':
            pred = (val_pred-max_min_satellite['pct37']['min'])/(max_min_satellite['pct37']['max']-max_min_satellite['pct37']['min'])
            target = (val_label-max_min_satellite['pct37']['min'])/(max_min_satellite['pct37']['max']-max_min_satellite['pct37']['min'])
        else:
            pred = val_pred
            target = val_label     
        mse = torch.mean((pred - target) ** 2)
        rmse += torch.sqrt(mse)
        
        
        #todo 加上绘制 renorm label 和 pred 的代码
        
        
        for thr in thresholds:
            has_event_target = (val_label >= thr)
            has_event_predict = (val_pred >= thr)
            
            hit = int(torch.sum(has_event_target & has_event_predict))
            miss = int(torch.sum(has_event_target & ~has_event_predict))
            false_alarm = int(torch.sum(~has_event_target & has_event_predict))
            no_event = int(torch.sum(~has_event_target))
        
            pod = hit / (hit + miss) if (hit + miss) > 0 else float(0)
            if pod >1:
                print(hit)
                print(hit + miss)
            far = false_alarm / no_event if no_event > 0 else float(0)
            if far > 1:
                print(false_alarm)
                print(no_event)
            csi = hit / (hit + miss + false_alarm) if (hit + miss + false_alarm) > 0 else float(0)
            
            total_pod[thr] += pod
            total_far[thr] += far
            total_csi[thr] += csi
    avg_rmse = rmse / (cal_num+1)
    avg_far_all = 0
    avg_pod_all = 0
    print('avg_rmse: ', avg_rmse)
    for thr in thresholds:
        avg_pod = total_pod[thr] / (cal_num)
        avg_far = total_far[thr] / (cal_num)
        avg_csi = total_csi[thr] / (cal_num)
        avg_far_all = avg_far+avg_far_all
        avg_pod_all = avg_pod+avg_pod_all
        print('thresholds: ', thr)
        print("avg_pod: ", avg_pod)
        print('avg_far: ', avg_far)
        print('avg_csi: ', avg_csi)
    # avg_far_all = avg_far_all/len(thresholds)
    # avg_pod_all = avg_pod_all/len(thresholds)
    # print('avg_far: ', avg_far_all)
    # print('avg_pod: ', avg_pod_all)

def compute_val_era5(out_var=4):
    if out_var == 3:
        folder_path = Path('/mnt/petrelfs/zhaoxiangyu1/code/weather_prompt_new/experiments/era5_t2m_2018/era5_3_12/val_test/Epoch1')
    else:
        folder_path = Path('/mnt/petrelfs/zhaoxiangyu1/code/weather_prompt_new/experiments/era5_u10_2018/era5_8_4/val_test/Epoch1')
    predict_files = []
    target_files = []
    for file in folder_path.glob('predict*'):
        predict_files.append(str(file))
    for file in folder_path.glob('target*'):
        target_files.append(str(file))
    predict_files.sort()
    target_files.sort()
    print(len(predict_files))
    assert len(predict_files)==len(target_files)
    rmse_score_sum = 0
    acc_score_sum = 0
    
    if out_var==3:
        clim = 0
        for i in range(len(predict_files)):
            val_label = (torch.from_numpy(np.load(target_files[i]))[64:192,:])* 21.32010786700973 + 278.7854495405721
            clim += val_label/len(predict_files)
        for idx in range(len(predict_files)):
            val_pred = (torch.from_numpy(np.load(predict_files[idx]))[64:192,:])* 21.32010786700973 + 278.7854495405721
            val_label = (torch.from_numpy(np.load(target_files[idx]))[64:192,:])* 21.32010786700973 + 278.7854495405721
            score = weighted_rmse_torch(val_pred, val_label)
            val_pred = val_pred-clim
            val_label = val_label-clim
            pred_prime = val_pred - torch.mean(val_pred)
            y_prime = val_label - torch.mean(val_label)
            score_acc = weighted_acc_torch(pred_prime, y_prime)
            rmse_score_sum += score
            acc_score_sum += score_acc
        # for idx in range(0, len(predict_files), 4):
        #     val_pred = (torch.from_numpy(np.load(predict_files[idx]))[64:192,:])* 21.32010786700973 + 278.7854495405721
        #     val_label = (torch.from_numpy(np.load(target_files[idx]))[64:192,:])* 21.32010786700973 + 278.7854495405721
        #     score = weighted_rmse_torch(val_pred, val_label)
        #     clim = torch.mean(val_label)
        #     score_acc = weighted_acc_torch(val_pred-clim, val_label-clim)
        #     rmse_score_sum += score
        #     acc_score_sum += score_acc
    elif out_var==8:
        clim = 0
        for i in range(len(predict_files)):
            val_label = (torch.from_numpy(np.load(target_files[i]))[64:192,:])* 3299.702929930327 + 53826.54542968748
            clim += val_label/len(predict_files)
        for idx in range(len(predict_files)):
            val_pred = (torch.from_numpy(np.load(predict_files[idx]))[64:192,:])*3299.702929930327 + 53826.54542968748
            val_label = (torch.from_numpy(np.load(target_files[idx]))[64:192,:])*3299.702929930327 + 53826.54542968748
            score = weighted_rmse_torch(val_pred, val_label)
            val_pred = val_pred-clim
            val_label = val_label-clim
            pred_prime = val_pred - torch.mean(val_pred)
            y_prime = val_label - torch.mean(val_label)
            score_acc = weighted_acc_torch(pred_prime, y_prime)
            rmse_score_sum += score
            acc_score_sum += score_acc
    else:
        clim=0
        for i in range(len(predict_files)):
            val_label = (torch.from_numpy(np.load(target_files[i]))[64:192,:])* 5.610453475051704 -0.14186215714480854
            clim += val_label/len(predict_files)

        for idx in range(len(predict_files)):

            val_pred = (torch.from_numpy(np.load(predict_files[idx]))[64:192,:])* 5.610453475051704 -0.14186215714480854
            
            val_label = (torch.from_numpy(np.load(target_files[idx]))[64:192,:])* 5.610453475051704 -0.14186215714480854

            score = weighted_rmse_torch(val_pred.unsqueeze(0), val_label.unsqueeze(0))
            val_pred = val_pred-clim
            val_label = val_label-clim
            pred_prime = val_pred - torch.mean(val_pred)
            y_prime = val_label - torch.mean(val_label)
            score_acc = weighted_acc_torch(pred_prime, y_prime)
            rmse_score_sum += score
            acc_score_sum += score_acc
    print('rmse:', rmse_score_sum)
    print('acc:', acc_score_sum)
    print('rmse:', rmse_score_sum/len(target_files))
    print('acc:', acc_score_sum/len(target_files))

    return rmse_score_sum/len(target_files)
        
compute_val_era5()
