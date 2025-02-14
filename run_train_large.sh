#!/bin/bash

while true
do 
  PORT=$((((RANDOM<<15)|RANDOM)%49152 + 10000))
  break
done
echo $PORT

srun -p ai4earth -n 1 --ntasks-per-node=1 --gres=gpu:8 --job-name=xiangyu --cpus-per-task=4 torchrun \
--nproc_per_node=8 --master_port $PORT train_PromptGIP.py \
--model mae_vit_large_patch16_dec512d8b_input256  \
--input_size 256 --batch_size 20 --mask_ratio 0.75 \
--warmup_epochs 1 --epochs 10 --blr 1e-4 \
--ckpt /mnt/petrelfs/zhaoxiangyu1/code/weather_prompt_new/experiments/large_finetune/8gpus/checkpoint-0.pth \
--save_ckpt_freq 1 \
--output_dir experiments/large_finetune/8gpus_new \
--data_path /mnt/petrelfs/zhaoxiangyu1/data/Test100_256 \
--data_path_val /mnt/petrelfs/zhaoxiangyu1/data/Test100_256 | tee -a experiments/PromptGIP_log.txt

