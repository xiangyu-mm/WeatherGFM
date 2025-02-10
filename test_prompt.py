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


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    cudnn.benchmark = True
    # 2 channels dataset
    rag_type = 'high'

    for id_num in range(5):

        dataset_val1 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='sevir',input_size=args.input_size,
                                                                    phase='val',task='sevir', RAG=rag_type, prompt_item=id_num)
        dataset_val2 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='ir_trans',input_size=args.input_size,
                                                                    phase='val',task='trans', RAG=rag_type, prompt_item=id_num)
        dataset_val3 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='inter',input_size=args.input_size,
                                                                    phase='val',task='inter', RAG=rag_type, prompt_item=id_num)
        dataset_val4 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='down_scaling_vil',input_size=args.input_size,
                                                                    phase='val',task='down_scaling', RAG=rag_type, prompt_item=id_num)
        dataset_val5 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='predict',input_size=args.input_size,
                                                                    phase='val',task='predict', RAG=rag_type, prompt_item=id_num)
        dataset_val6 = lowlevel_prompt_dataloader.Dataset_dblur(split='val', prompt='random', prompt_id=id_num)

        print(dataset_val1.__len__())

        # single | mix | mismatch_single | UDC_Toled | UDC_Poled | Rain100L | Rain100H | Test100 | Test1200
        # SOTS | mismatch_mix
        # dataset_val_single = lowlevel_prompt_dataloader.DatasetLowlevel_Customized_Test_DirectLoad_Triplet(data_path)

        # data_loader_train_4c = None
        
        data_loader_val1 = torch.utils.data.DataLoader(
            dataset_val1,
            batch_size=12,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        data_loader_val2 = torch.utils.data.DataLoader(
            dataset_val2,
            batch_size=12,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        
        data_loader_val3 = torch.utils.data.DataLoader(
            dataset_val3,
            batch_size=12,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        data_loader_val4 = torch.utils.data.DataLoader(
            dataset_val4,
            batch_size=12,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        data_loader_val5 = torch.utils.data.DataLoader(
            dataset_val5,
            batch_size=4,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        data_loader_val6 = torch.utils.data.DataLoader(
            dataset_val6,
            batch_size=4,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        ) 

        # define the model
        model = models_mae_PromptGIP_CNN_Head.__dict__[args.model]()
        
        if args.ckpt:
            print('Load pretrained model: ', args.ckpt)
            checkpoint = torch.load(args.ckpt, map_location='cpu')
            msg = model.load_state_dict(checkpoint['model'], strict=False)
            print(msg)

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
        rag_type = 'random'
        val_model_CNN_Head(model_without_ddp.cuda(), data_loader_val5, 10, args.output_dir+'/'+rag_type+'_prompt/'+str(id_num), mode='val_test',
                                patch_size=16, data_type='predict')
        # val_model_CNN_Head(model_without_ddp.cuda(), data_loader_val1, 10, args.output_dir+'/'+rag_type+'_prompt/'+str(id_num), mode='val_test',
        #                         patch_size=16, data_type='sevir')
        # val_model_CNN_Head(model_without_ddp.cuda(), data_loader_val2, 10, args.output_dir+'/'+rag_type+'_prompt/'+str(id_num), mode='val_test', 
        #                         patch_size=16, data_type='trans')
        # val_model_CNN_Head(model_without_ddp.cuda(), data_loader_val3, 10, args.output_dir+'/'+rag_type+'_prompt/'+str(id_num), mode='val_test', 
        #                         patch_size=16, data_type='inter')
        # val_model_CNN_Head(model_without_ddp.cuda(), data_loader_val4, 10, args.output_dir+'/'+rag_type+'_prompt/'+str(id_num), mode='val_test', 
        #                         patch_size=16, data_type='down_scaling_vil')
        # val_model_CNN_Head(model_without_ddp.cuda(), data_loader_val6, 10, args.output_dir+'/'+rag_type+'_prompt/'+str(id_num), mode='val_test', 
        #                         patch_size=16, data_type='dblur')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
