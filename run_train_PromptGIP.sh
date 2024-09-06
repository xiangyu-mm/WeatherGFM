nohup srun -p ai4earth -n 1 --ntasks-per-node=1 --gres=gpu:8 --job-name=xiangyu --cpus-per-task=8 torchrun \
--nproc_per_node=2 --master_port=4356 train_PromptGIP.py \
--model mae_vit_large_patch16_dec512d8b_input256 \
--input_size 256 --batch_size 20 --mask_ratio 0.75 \
--warmup_epochs 1 --epochs 5 --blr 1e-4 \
--save_ckpt_freq 5 \
--ckpt /mnt/petrelfs/zhaoxiangyu1/code/weather_prompt_new/experiments/weather_5tasks/checkpoint-40.pth \
--output_dir experiments/weather_10taskss \
--data_path /mnt/petrelfs/zhaoxiangyu1/data/Test100_256 \
--data_path_val /mnt/petrelfs/zhaoxiangyu1/data/Test100_256 | tee -a experiments/PromptGIP_log.txt \
> myout_5t_8gpus.log 2>&1 &

# srun -p ai4earth -n 1 --ntasks-per-node=1 --gres=gpu:1 --job-name=genlv100 --cpus-per-task=16 torchrun \
# --master_port 31956 000_main_only_train_GenLV100_from_scratch.py --model xrestormer_prompt_crossattn_large \
# --input_size 256 --num_workers 16 --batch_size 4 --warmup_epochs 0 --start_epoch 0 --epochs 100 --blr 1e-4 \
# --save_ckpt_freq 1 --output_dir experiments/003_main_only_train_GenLV100_from_scratch_large \
# --data_path_val /nvme/liuyihao/DATA/Common528/Image_clean_256x256 | tee -a experiments/PromptGIP_log.txt

