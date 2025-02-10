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
from evaluation_prompt import *
from dataset.rainnet_dataloader import RainNetDataset
from dataset.satellite_dataloader import MyDataset
from dataset.apd_dataloader import APDataset

from petrel_client.client import Client
from dataset.era5_dataloader import era5_processed

client = Client(conf_path="~/petreloss.conf")


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--notfinetune', action='store_false')
    parser.set_defaults(notfinetune=True)

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
    args.second_input_size = 224
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    config = load_config("./config/xrestormer.yaml")
    # 2 channels dataset
    dataset_train1 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='sevir',input_size=args.input_size,
                                                                   phase='train',task='sevir')
    dataset_train2 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='ir_trans',input_size=args.input_size,
                                                                   phase='train',task='trans', not_finetune=args.notfinetune)
    dataset_train3 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='inter',input_size=args.input_size,
                                                                   phase='train',task='inter', not_finetune=args.notfinetune)
    dataset_train4 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='down_scaling_vil',input_size=args.input_size,
                                                                   phase='train',task='down_scaling', not_finetune=args.notfinetune)
    dataset_train5 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='vis_recon',input_size=args.input_size,
                                                                   phase='train',task='down_scaling', not_finetune=args.notfinetune)
    dataset_train6 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='down_scaling_ir',input_size=args.input_size,
                                                                   phase='train',task='down_scaling', not_finetune=args.notfinetune)
    dataset_train7 = lowlevel_prompt_dataloader.Dataset_dblur(split='train')
    # 4 channels dataset
    dataset_train_4c1 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='ir_predict',input_size=args.input_size,
                                                                   phase = 'train',task='predict', not_finetune=args.notfinetune)
    dataset_train_4c2 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='predict',input_size=args.input_size,
                                                                   phase = 'train',task='predict')
    # NO2_dataset
    dataset_train9 = APDataset(split='train')

    dataset_train_2c = dataset_train1+dataset_train2+dataset_train3+dataset_train4+dataset_train5+\
                        dataset_train6+dataset_train7+dataset_train9
    dataset_train_4c = era5_processed(split='train', out_var=3)

    dataset_era5 = era5_processed(split='train', out_var=8, continue_time=True)
    print(dataset_train_2c.__len__())
    # dataset_train_2c = dataset_train_4c
    print(dataset_train_4c.__len__())
    
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
    dataset_val7 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='ir_predict',input_size=args.input_size,
                                                                 phase='val',task='ir_predict')
    dataset_val8 = lowlevel_prompt_dataloader.DatasetSevir_Train(data_path='vis_recon',input_size=args.input_size,
                                                                 phase='val',task='vis_recon')
    dataset_val9 = lowlevel_prompt_dataloader.Dataset_dblur(split='val')

    print(dataset_val1.__len__())

    # single | mix | mismatch_single | UDC_Toled | UDC_Poled | Rain100L | Rain100H | Test100 | Test1200
    # SOTS | mismatch_mix
    test_folder = 'input' 
    data_path = os.path.join('/mnt/petrelfs/zhaoxiangyu1/data/Test100_256', test_folder)
    # dataset_val_single = lowlevel_prompt_dataloader.DatasetLowlevel_Customized_Test_DirectLoad_Triplet(data_path)
    
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train_2c, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_train_4c = torch.utils.data.DistributedSampler(
            dataset_train_4c, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_train_era5 = torch.utils.data.DistributedSampler(
            dataset_era5, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train_2c, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_train_4c = torch.utils.data.DataLoader(
        dataset_train_4c, sampler=sampler_train_4c,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_era5 = torch.utils.data.DataLoader(
        dataset_era5, sampler=sampler_train_era5,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # data_loader_train_4c = None
    
    data_loader_val1 = torch.utils.data.DataLoader(
        dataset_val1,
        batch_size=24,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    data_loader_val2 = torch.utils.data.DataLoader(
        dataset_val2,
        batch_size=24,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    data_loader_val3 = torch.utils.data.DataLoader(
        dataset_val3,
        batch_size=24,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    data_loader_val4 = torch.utils.data.DataLoader(
        dataset_val4,
        batch_size=24,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    data_loader_val5 = torch.utils.data.DataLoader(
        dataset_val5,
        batch_size=24,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    data_loader_val6 = torch.utils.data.DataLoader(
        dataset_val6,
        batch_size=24,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    data_loader_val7 = torch.utils.data.DataLoader(
        dataset_val7,
        batch_size=24,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val8 = torch.utils.data.DataLoader(
        dataset_val8,
        batch_size=24,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val9 = torch.utils.data.DataLoader(
        dataset_val9,
        batch_size=24,
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
    # epoch_size = len(dataset_train_2c)+len(dataset_train_4c)
    epoch_size = len(dataset_era5)
    
    print(f'epoch_size is {epoch_size}')
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    base_lr = (args.lr * 256 / eff_batch_size)
    print("base lr: %.2e" % base_lr)
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    for k, v in model_without_ddp.named_parameters():
        if 'vae' in k:
            v.requires_grad = False

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_era5.sampler.set_epoch(epoch)
            if data_loader_train_4c is not None:
                data_loader_train_4c.sampler.set_epoch(epoch)

        train_one_epoch(
            model, data_loader_era5,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            epoch_size=epoch_size // eff_batch_size * args.accum_iter,
            data_loader_appendix=None
        )

        if args.output_dir and (epoch % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        # val_on_master_CNN_Head(model_without_ddp.cuda(), data_loader_val8, epoch, args.output_dir, mode='val_test', 
        #                        patch_size=16, data_type='vis_recon')
        # val_on_master_CNN_Head(model_without_ddp.cuda(), data_loader_val3, epoch, args.output_dir, mode='val_test', 
        #                        patch_size=16, data_type='inter')
        # val_on_master_CNN_Head(model_without_ddp.cuda(), data_loader_val_single, epoch, args.output_dir, mode='val_test_single', patch_size=16)
        # val_on_master_CNN_Head(model_without_ddp.cuda(), data_loader_val_mix, epoch, args.output_dir, mode='val_test_mix', patch_size=16)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # if misc.is_main_process():
    #     run.finish()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
