#!/bin/bash

while true
do 
  PORT=$((((RANDOM<<15)|RANDOM)%49152 + 10000))
  break
done
echo $PORT

srun -p ai4earth -n 1 --ntasks-per-node=1 --gres=gpu:8 --job-name=xiangyu --cpus-per-task=4 torchrun \
--nproc_per_node=8 --master_port $PORT train_PromptGIP.py \
--model mae_vit_base_patch16_dec512d8b  \
--input_size 256 --batch_size 40 --mask_ratio 0.75 \
--warmup_epochs 1 --epochs 20 --blr 1e-4 \
--ckpt /mnt/petrelfs/zhaoxiangyu1/code/weather_prompt_new/experiments/base_strach/8gpus/checkpoint-0.pth \
--save_ckpt_freq 1 \
--notfinetune \
--output_dir experiments/base_strach/all_tasks \
--data_path /mnt/petrelfs/zhaoxiangyu1/data/Test100_256 \
--data_path_val /mnt/petrelfs/zhaoxiangyu1/data/Test100_256 | tee -a experiments/PromptGIP_log.txt \

# srun -p ai4earth -n 1 --ntasks-per-node=1 --gres=gpu:1 --job-name=genlv100 --cpus-per-task=16 torchrun \
# --master_port 31956 000_main_only_train_GenLV100_from_scratch.py --model xrestormer_prompt_crossattn_large \
# --input_size 256 --num_workers 16 --batch_size 4 --warmup_epochs 0 --start_epoch 0 --epochs 100 --blr 1e-4 \
# --save_ckpt_freq 1 --output_dir experiments/003_main_only_train_GenLV100_from_scratch_large \
# --data_path_val /nvme/liuyihao/DATA/Common528/Image_clean_256x256 | tee -a experiments/PromptGIP_log.txt
# --ckpt /mnt/petrelfs/zhaoxiangyu1/code/weather_prompt_new/experiments/weather_5tasks/checkpoint-40.pth \

