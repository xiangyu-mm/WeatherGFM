import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision import datasets
import timm.optim.optim_factory as optim_factory
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_mae_PromptGIP_CNN_Head
from engine_pretrain import train_one_epoch
from dataset.data_utiles import load_config
import dataset.lowlevel_PromtGIP_dataloader as lowlevel_prompt_dataloader
from dataset.lowlevel_PromtGIP_dataloader import Dataset_dblur
from evaluation_prompt import *
from dataset.rainnet_dataloader import RainNetDataset
from dataset.satellite_dataloader import MyDataset
from dataset.apd_dataloader import APDataset
from dataset.era5_dataloader import era5_processed

from petrel_client.client import Client

client = Client(conf_path="~/petreloss.conf")


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=4, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--ckpt', help='resume from checkpoint')
    
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--break_after_epoch', type=int, metavar='N', help='break training after X epochs, to tune hyperparams and avoid messing with training schedule')


    # Dataset parameters
    parser.add_argument('--data_path', default='/shared/yossi_gandelsman/arxiv/arxiv_data/', type=str,
                        help='dataset path')
    parser.add_argument('--data_path_val', default='/shared/yossi_gandelsman/arxiv/arxiv_data/', type=str,
                        help='val dataset path')
    parser.add_argument('--imagenet_percent', default=1, type=float)
    parser.add_argument('--subsample', action='store_true')
    parser.set_defaults(subsample=False)
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    cudnn.benchmark = True
    # 2 channels dataset
    
    dataset_val1 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='sevir',input_size=args.input_size,
                                                                 phase='val',task='sevir')
    dataset_val2 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='ir_trans',input_size=args.input_size,
                                                                 phase='val',task='trans')
    dataset_val3 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='inter',input_size=args.input_size,
                                                                 phase='val',task='inter')
    dataset_val4 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='down_scaling_vil',input_size=args.input_size,
                                                                 phase='val',task='down_scaling')
    dataset_val5 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='predict',input_size=args.input_size,
                                                                 phase='val',task='predict')
    dataset_val6 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='down_scaling_ir',input_size=args.input_size,
                                                                 phase='val',task='down_scaling')
    dataset_val9 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='vis_recon',input_size=args.input_size,
                                                                 phase='val',task='vis_recon')
    dataset_val10 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='ir_predict',input_size=args.input_size,
                                                                 phase='val',task='ir_predict')
    dataset_val7 = RainNetDataset('val')
    dataset_val8 = MyDataset(dataset_type='val', split='val')
    dataset_val11 = APDataset(split='val')
    dataset_val12 = Dataset_dblur(split='val')

    #OOD val dataset
    dataset_val13 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='ir_trans',input_size=args.input_size,
                                                                 phase='val',task='ir107_trans_ir069')
    dataset_val14 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='vis_predict',input_size=args.input_size,
                                                                 phase='val',task='vis_predict')
    dataset_val15 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='ir107_predict',input_size=args.input_size,
                                                                 phase='val',task='ir107_predict')
    dataset_val16 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='down_scaling_ir107',input_size=args.input_size,
                                                                 phase='val',task='down_scaling_ir107')
    dataset_val17 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='multi_downscaling',input_size=args.input_size,
                                                                 phase='val',task='multi_downscaling')
    dataset_val18 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='sevir_vis',input_size=args.input_size,
                                                                 phase='val',task='sevir_vis')
    dataset_val19 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='down_scaling_vil_8x',input_size=args.input_size,
                                                                 phase='val',task='down_scaling_vil_8x')
    

    dataset_val20 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='vil_inter_15min',input_size=args.input_size,
                                                                 phase='val',task='vil_inter_15min')
    dataset_val21 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='vil_inter_60min',input_size=args.input_size,
                                                                 phase='val',task='vil_inter_60min')
    dataset_val22 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='ir069_inter_60min',input_size=args.input_size,
                                                                 phase='val',task='ir069_inter_60min')
    
    # era5 dataset lead_time = [1,4,12,20,28]
    lead_times = [1,4,12,20,28]
    for lead_time in lead_times:
        dataset_val23 = era5_processed(split='valid', out_var=3, lead_time=lead_time)
        dataset_val24 = era5_processed(split='valid', out_var=8, lead_time=lead_time)
        dataset_val25 = era5_processed(split='valid', out_var=4, lead_time=lead_time, continue_time=False)


        # single | mix | mismatch_single | UDC_Toled | UDC_Poled | Rain100L | Rain100H | Test100 | Test1200
        # SOTS | mismatch_mix
        # dataset_val_single = lowlevel_prompt_dataloader.DatasetLowlevel_Customized_Test_DirectLoad_Triplet(data_path)

        # data_loader_train_4c = None
        
        data_loader_val1 = torch.utils.data.DataLoader(
            dataset_val1,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        data_loader_val2 = torch.utils.data.DataLoader(
            dataset_val2,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        
        data_loader_val3 = torch.utils.data.DataLoader(
            dataset_val3,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        data_loader_val4 = torch.utils.data.DataLoader(
            dataset_val4,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        data_loader_val5 = torch.utils.data.DataLoader(
            dataset_val5,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        data_loader_val6 = torch.utils.data.DataLoader(
            dataset_val6,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        data_loader_val7 = torch.utils.data.DataLoader(
            dataset_val7,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        data_loader_val8 = torch.utils.data.DataLoader(
            dataset_val8,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        data_loader_val9 = torch.utils.data.DataLoader(
            dataset_val9,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        data_loader_val10 = torch.utils.data.DataLoader(
            dataset_val10,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        data_loader_val11 = torch.utils.data.DataLoader(
            dataset_val11,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        data_loader_val12 = torch.utils.data.DataLoader(
            dataset_val12,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        data_loader_val13 = torch.utils.data.DataLoader(
            dataset_val13,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        data_loader_val14 = torch.utils.data.DataLoader(
            dataset_val14,
            batch_size=12,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        data_loader_val15 = torch.utils.data.DataLoader(
            dataset_val15,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        data_loader_val16 = torch.utils.data.DataLoader(
            dataset_val16,
            batch_size=12,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        data_loader_val17 = torch.utils.data.DataLoader(
            dataset_val17,
            batch_size=12,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        data_loader_val18 = torch.utils.data.DataLoader(
            dataset_val18,
            batch_size=12,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        data_loader_val19 = torch.utils.data.DataLoader(
            dataset_val19,
            batch_size=12,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        data_loader_val20 = torch.utils.data.DataLoader(
            dataset_val20,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        data_loader_val21 = torch.utils.data.DataLoader(
            dataset_val21,
            batch_size=12,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        data_loader_val22 = torch.utils.data.DataLoader(
            dataset_val22,
            batch_size=12,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        data_loader_val23 = torch.utils.data.DataLoader(
            dataset_val23,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        data_loader_val24 = torch.utils.data.DataLoader(
            dataset_val24,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        data_loader_val25 = torch.utils.data.DataLoader(
            dataset_val25,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        print('start testing')
        # define the model
        model = models_mae_PromptGIP_CNN_Head.__dict__[args.model]()
        
        if args.ckpt:
            print('Load pretrained model: ', args.ckpt)
            checkpoint = torch.load(args.ckpt, map_location='cpu')
            msg = model.load_state_dict(checkpoint['model'], strict=False)
            print(msg)
            print(f"模型参数量: {count_parameters(model)}")

        model.to(device)
        
        model_without_ddp = model
        print("Model = %s" % str(model_without_ddp))

        # if args.distributed:
        #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        #     model_without_ddp = model.module
        
        # following timm: set wd as 0 for bias and norm layers
        for k, v in model_without_ddp.named_parameters():
            if 'vae' in k:
                v.requires_grad = False

        # val_model_CNN_Head(model_without_ddp.cuda(), data_loader_val23, 1, args.output_dir+'/'+'era5_3_'+str(lead_time), mode='val_test', 
        #                         patch_size=16, data_type='era5', val_mode=True)
        val_model_CNN_Head(model_without_ddp.cuda(), data_loader_val25, 1, args.output_dir+'/'+'era5_8_'+str(lead_time), mode='val_test', 
                                patch_size=16, data_type='era5', val_mode=True)
        # val_model_CNN_Head(model_without_ddp.cuda(), data_loader_val24, 1, args.output_dir+'/'+'era5_8_'+str(lead_time), mode='val_test', 
        #                         patch_size=16, data_type='era5', val_mode=True)
        # val_model_CNN_Head(model_without_ddp.cuda(), data_loader_val1, 1, args.output_dir, mode='val_test', 
        #                         patch_size=16, data_type='sevir', val_mode=True)
        # val_model_CNN_Head(model_without_ddp.cuda(), data_loader_val2, 1, args.output_dir, mode='val_test', 
        #                         patch_size=16, data_type='trans', val_mode=True)
        # val_model_CNN_Head(model_without_ddp.cuda(), data_loader_val3, 1, args.output_dir, mode='val_test', 
        #                         patch_size=16, data_type='inter', val_mode=True)
        # val_model_CNN_Head(model_without_ddp.cuda(), data_loader_val13, 1, args.output_dir, mode='val_test', 
        #                         patch_size=16, data_type='ir107_trans_ir069', val_mode=True)
        # val_model_CNN_Head(model_without_ddp.cuda(), data_loader_val15, 1, args.output_dir, mode='val_test', 
        #                         patch_size=16, data_type='ir107_predict', val_mode=True)
        # val_model_CNN_Head(model_without_ddp.cuda(), data_loader_val20, 1, args.output_dir, mode='val_test', 
        #                         patch_size=16, data_type='vil_inter_15min', val_mode=True)




if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
