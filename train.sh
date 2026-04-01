### training TTSR
#rmdir /s /q .\train\CUSTOM\TTSR
/d/anaconda3/envs/pmypytorch2/python.exe main.py \
               --save_dir ./train/CUSTOM/TTSR \
               --reset True \
               --log_file_name train.log \
               --num_gpu 1 \
               --num_workers 4 \
               --dataset CUSTOM \
               --dataset_dir ./CUSTOM \
               --n_feats 64 \
               --lr_rate 1e-4 \
               --lr_rate_dis 1e-4 \
               --lr_rate_lte 1e-5 \
               --rec_w 1 \
               --per_w 1e-2 \
               --tpl_w 1e-2 \
               --adv_w 1e-3 \
               --batch_size 4 \
               --train_crop_size 40 \
               --num_init_epochs 2 \
               --num_epochs 50 \
               --print_every 50 \
               --save_every 10 \
               --val_every 10

#/d/anaconda3/envs/pmypytorch2/python.exe main.py \
#               --save_dir ./train_self/CUFED/TTSR \
#               --reset True \
#               --log_file_name train.log \
#               --num_gpu 1 \
#               --num_workers 9 \
#               --dataset CUFED \
#               --dataset_dir ./CUFED \
#               --n_feats 64 \
#               --lr_rate 1e-4 \
#               --lr_rate_dis 1e-4 \
#               --lr_rate_lte 1e-5 \
#               --rec_w 1 \
#               --per_w 1e-2 \
#               --tpl_w 1e-2 \
#               --adv_w 1e-3 \
#               --batch_size 2 \
#               --num_workers 0 \
#               --num_init_epochs 2 \
#               --num_epochs 150 \
#               --print_every 600 \
#               --save_every 10 \
#               --val_every 10 \
#               --model_path ./TTSR.pt    # 使用作者的预训练权重




# ### training TTSR-rec
# python main.py --save_dir ./train/CUFED/TTSR-rec \
#                --reset True \
#                --log_file_name train.log \
#                --num_gpu 1 \
#                --num_workers 9 \
#                --dataset CUFED \
#                --dataset_dir /home/v-fuyang/Data/CUFED/ \
#                --n_feats 64 \
#                --lr_rate 1e-4 \
#                --lr_rate_dis 1e-4 \
#                --lr_rate_lte 1e-5 \
#                --rec_w 1 \
#                --per_w 0 \
#                --tpl_w 0 \
#                --adv_w 0 \
#                --batch_size 9 \
#                --num_init_epochs 0 \
#                --num_epochs 200 \
#                --print_every 600 \
#                --save_every 10 \
#                --val_every 10
