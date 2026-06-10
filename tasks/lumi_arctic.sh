#!/bin/bash
#SBATCH --job-name=Train7
#SBATCH --account=project_465002698
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=small-g
#SBATCH --output=/scratch/project_465002698/venky/projects/arctic/out_files/%x_%j.out
#SBATCH --error=/scratch/project_465002698/venky/projects/arctic/out_files/%x_%j.err
#SBATCH --time=48:00:00

hostname
rocm-smi
echo $CUDA_VISIBLE_DEVICES

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword elu_bw15_split1 --dataset geofoldsv1 --use_wb --data_fold 1 &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword elu_bw15_split2 --dataset geofoldsv1 --use_wb --data_fold 2 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --keyword elu_bw15_split3 --dataset geofoldsv1 --use_wb --data_fold 3 &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --keyword elu_bw15_split4 --dataset geofoldsv1 --use_wb --data_fold 4 &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --keyword elu_bw15_split5 --dataset geofoldsv1 --use_wb --data_fold 5 &
#ROCR_VISIBLE_DEVICES=5 python train_arctic.py --keyword elu_bw15_split6 --dataset geofoldsv2 --use_wb --data_fold 1 &
#ROCR_VISIBLE_DEVICES=6 python train_arctic.py --keyword elu_bw15_split7 --dataset geofoldsv2 --use_wb --data_fold 2 &
#ROCR_VISIBLE_DEVICES=7 python train_arctic.py --keyword elu_bw15_split8 --dataset geofoldsv2 --use_wb --data_fold 3 &
#wait

ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword elu_bw15_split9 --dataset geofoldsv2 --use_wb --data_fold 4 &
ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword elu_bw15_split10 --dataset geofoldsv2 --use_wb --data_fold 5 &
wait

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword split1_elu_lr601 --use_wb  --learning_rate 0.000001 &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword split1_elu_lr603 --use_wb  --learning_rate 0.000003 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --keyword split1_elu_lr607 --use_wb  --learning_rate 0.000007 &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --keyword split1_elu_lr609 --use_wb  --learning_rate 0.000009 &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --keyword split1_elu_lr701 --use_wb  --learning_rate 0.0000001 &
#wait

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword split1_elu_lr101 --use_wb  --learning_rate 0.1 &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword split1_elu_lr105 --use_wb  --learning_rate 0.5 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --keyword split1_elu_lr205 --use_wb  --learning_rate 0.05 &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --keyword split1_elu_lr305 --use_wb  --learning_rate 0.005 &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --keyword split1_elu_lr505 --use_wb  --learning_rate 0.00005 &
#ROCR_VISIBLE_DEVICES=5 python train_arctic.py --keyword split1_elu_lr605 --use_wb  --learning_rate 0.000005 &
#wait

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --dataset geofoldsv1 --keyword bw15_split1_elu --use_wb --data_fold 1 --model_name unet_elu &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --dataset geofoldsv1 --keyword bw15_split2_elu --use_wb --data_fold 2 --model_name unet_elu &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --dataset geofoldsv1 --keyword bw15_split3_elu --use_wb --data_fold 3 --model_name unet_elu &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --dataset geofoldsv1 --keyword bw15_split4_elu --use_wb --data_fold 4 --model_name unet_elu &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --dataset geofoldsv1 --keyword bw15_split5_elu --use_wb --data_fold 5 --model_name unet_elu &
#ROCR_VISIBLE_DEVICES=5 python train_arctic.py --dataset geofoldsv2 --keyword bw15_split6_elu --use_wb --data_fold 1 --model_name unet_elu &
#ROCR_VISIBLE_DEVICES=6 python train_arctic.py --dataset geofoldsv2 --keyword bw15_split7_elu --use_wb --data_fold 2 --model_name unet_elu &
#ROCR_VISIBLE_DEVICES=7 python train_arctic.py --dataset geofoldsv2 --keyword bw15_split8_elu --use_wb --data_fold 3 --model_name unet_elu &
#wait

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --dataset geofoldsv1 --keyword bw15_split1_elu_bn --use_wb --data_fold 1 --model_name unet_elu_bn &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --dataset geofoldsv1 --keyword bw15_split2_elu_bn --use_wb --data_fold 2 --model_name unet_elu_bn &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --dataset geofoldsv1 --keyword bw15_split3_elu_bn --use_wb --data_fold 3 --model_name unet_elu_bn &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --dataset geofoldsv1 --keyword bw15_split4_elu_bn --use_wb --data_fold 4 --model_name unet_elu_bn &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --dataset geofoldsv1 --keyword bw15_split5_elu_bn --use_wb --data_fold 5 --model_name unet_elu_bn &
#ROCR_VISIBLE_DEVICES=5 python train_arctic.py --dataset geofoldsv2 --keyword bw15_split6_elu_bn --use_wb --data_fold 1 --model_name unet_elu_bn &
#ROCR_VISIBLE_DEVICES=6 python train_arctic.py --dataset geofoldsv2 --keyword bw15_split7_elu_bn --use_wb --data_fold 2 --model_name unet_elu_bn &
#ROCR_VISIBLE_DEVICES=7 python train_arctic.py --dataset geofoldsv2 --keyword bw15_split8_elu_bn --use_wb --data_fold 3 --model_name unet_elu_bn &
#wait

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --dataset geofoldsv2 --keyword bw15_split9_elu --use_wb --data_fold 4 --model_name unet_elu &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --dataset geofoldsv2 --keyword bw15_split10_elu --use_wb --data_fold 5 --model_name unet_elu &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --dataset geofoldsv2 --keyword bw15_split9_elu_bn --use_wb --data_fold 4 --model_name unet_elu_bn &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --dataset geofoldsv2 --keyword bw15_split10_elu_bn --use_wb --data_fold 5 --model_name unet_elu_bn &
#wait


#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --dataset geofoldsv1 --keyword bw5_split1 --use_wb --data_fold 1 --boundary_weight 5 &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --dataset geofoldsv1 --keyword bw5_split2 --use_wb --data_fold 2 --boundary_weight 5 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --dataset geofoldsv1 --keyword bw5_split3 --use_wb --data_fold 3 --boundary_weight 5 &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --dataset geofoldsv1 --keyword bw5_split4 --use_wb --data_fold 4 --boundary_weight 5 &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --dataset geofoldsv1 --keyword bw5_split5 --use_wb --data_fold 5 --boundary_weight 5 &
#ROCR_VISIBLE_DEVICES=5 python train_arctic.py --dataset geofoldsv2 --keyword bw5_split6 --use_wb --data_fold 1 --boundary_weight 5 &
#ROCR_VISIBLE_DEVICES=6 python train_arctic.py --dataset geofoldsv2 --keyword bw5_split7 --use_wb --data_fold 2 --boundary_weight 5 &
#ROCR_VISIBLE_DEVICES=7 python train_arctic.py --dataset geofoldsv2 --keyword bw5_split8 --use_wb --data_fold 3 --boundary_weight 5 &
#wait

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --dataset geofoldsv1 --keyword bw1_split1 --use_wb --data_fold 1 &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --dataset geofoldsv1 --keyword bw1_split2 --use_wb --data_fold 2 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --dataset geofoldsv1 --keyword bw1_split3 --use_wb --data_fold 3 &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --dataset geofoldsv1 --keyword bw1_split4 --use_wb --data_fold 4 &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --dataset geofoldsv1 --keyword bw1_split5 --use_wb --data_fold 5 &
#ROCR_VISIBLE_DEVICES=5 python train_arctic.py --dataset geofoldsv2 --keyword bw1_split6 --use_wb --data_fold 1 &
#ROCR_VISIBLE_DEVICES=6 python train_arctic.py --dataset geofoldsv2 --keyword bw1_split7 --use_wb --data_fold 2 &
#ROCR_VISIBLE_DEVICES=7 python train_arctic.py --dataset geofoldsv2 --keyword bw1_split8 --use_wb --data_fold 3 &
#wait

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --dataset geofoldsv2 --keyword bw1_split9 --use_wb --data_fold 4 &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --dataset geofoldsv2 --keyword bw1_split10 --use_wb --data_fold 5 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --dataset geofoldsv2 --keyword bw5_split9 --use_wb --data_fold 4 --boundary_weight 5 &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --dataset geofoldsv2 --keyword bw5_split10 --use_wb --data_fold 5 --boundary_weight 5 &
#wait

#ROCR_VISIBLE_DEVICES=0 python pred_tiles.py --patch_size 4096 --chunk_id 0 &
#ROCR_VISIBLE_DEVICES=1 python pred_tiles.py --patch_size 4096 --chunk_id 1 &
#ROCR_VISIBLE_DEVICES=2 python pred_tiles.py --patch_size 4096 --chunk_id 2 &
#ROCR_VISIBLE_DEVICES=3 python pred_tiles.py --patch_size 4096 --chunk_id 3 &
#ROCR_VISIBLE_DEVICES=4 python pred_tiles.py --patch_size 4096 --chunk_id 4 &
#ROCR_VISIBLE_DEVICES=5 python pred_tiles.py --patch_size 4096 --chunk_id 5 &
#ROCR_VISIBLE_DEVICES=6 python pred_tiles.py --patch_size 4096 --chunk_id 6 &
#ROCR_VISIBLE_DEVICES=7 python pred_tiles.py --patch_size 4096 --chunk_id 7 &
#wait

#ROCR_VISIBLE_DEVICES=0 python pred_tiles.py --patch_size 4096 --chunk_id 8 &
#ROCR_VISIBLE_DEVICES=1 python pred_tiles.py --patch_size 4096 --chunk_id 9 &
#ROCR_VISIBLE_DEVICES=2 python pred_tiles.py --patch_size 4096 --chunk_id 10 &
#ROCR_VISIBLE_DEVICES=3 python pred_tiles.py --patch_size 4096 --chunk_id 11 &
#ROCR_VISIBLE_DEVICES=4 python pred_tiles.py --patch_size 4096 --chunk_id 12 &
#ROCR_VISIBLE_DEVICES=5 python pred_tiles.py --patch_size 4096 --chunk_id 13 &
#ROCR_VISIBLE_DEVICES=6 python pred_tiles.py --patch_size 4096 --chunk_id 14 &
#ROCR_VISIBLE_DEVICES=7 python pred_tiles.py --patch_size 4096 --chunk_id 15 &
#wait

#ROCR_VISIBLE_DEVICES=0 python pred_tiles.py --patch_size 4096 --chunk_id 16 &
#ROCR_VISIBLE_DEVICES=1 python pred_tiles.py --patch_size 4096 --chunk_id 17 &
#ROCR_VISIBLE_DEVICES=2 python pred_tiles.py --patch_size 4096 --chunk_id 18 &
#ROCR_VISIBLE_DEVICES=3 python pred_tiles.py --patch_size 4096 --chunk_id 19 &
#ROCR_VISIBLE_DEVICES=4 python pred_tiles.py --patch_size 4096 --chunk_id 20 &
#ROCR_VISIBLE_DEVICES=5 python pred_tiles.py --patch_size 4096 --chunk_id 21 &
#ROCR_VISIBLE_DEVICES=6 python pred_tiles.py --patch_size 4096 --chunk_id 22 &
#ROCR_VISIBLE_DEVICES=7 python pred_tiles.py --patch_size 4096 --chunk_id 23 &
#wait

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword bw15_new_fold1 --use_wb --data_fold 1 &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword bw15_new_fold2 --use_wb --data_fold 2 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --keyword bw15_new_fold3 --use_wb --data_fold 3 &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --keyword bw15_new_fold4 --use_wb --data_fold 4 &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --keyword bw15_new_fold5 --use_wb --data_fold 5 &
#wait

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword Sbw10_all --use_wb --data_fold 1 --boundary_weight 10 &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword Sbw20_all --use_wb --data_fold 1 --boundary_weight 20 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --keyword all --use_wb --data_fold 1 &
#wait


#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword Sbw10_fold1 --use_wb --data_fold 1 --boundary_weight 10 &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword Sbw10_fold2 --use_wb --data_fold 2 --boundary_weight 10 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --keyword Sbw10_fold3 --use_wb --data_fold 3 --boundary_weight 10 &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --keyword Sbw10_fold4 --use_wb --data_fold 4 --boundary_weight 10 &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --keyword Sbw10_fold5 --use_wb --data_fold 5 --boundary_weight 10 &
#ROCR_VISIBLE_DEVICES=5 python train_arctic.py --keyword Sbw20_fold1 --use_wb --data_fold 1 --boundary_weight 20 &
#ROCR_VISIBLE_DEVICES=6 python train_arctic.py --keyword Sbw20_fold2 --use_wb --data_fold 2 --boundary_weight 20 &
#ROCR_VISIBLE_DEVICES=7 python train_arctic.py --keyword Sbw20_fold3 --use_wb --data_fold 3 --boundary_weight 20 &
#wait

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword Sbw20_fold4 --use_wb --data_fold 4 --boundary_weight 20 &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword Sbw20_fold5 --use_wb --data_fold 5 --boundary_weight 20 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --keyword Sbw50_fold1 --use_wb --data_fold 1 --boundary_weight 50 &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --keyword Sbw50_fold2 --use_wb --data_fold 2 --boundary_weight 50 &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --keyword Sbw50_fold3 --use_wb --data_fold 3 --boundary_weight 50 &
#ROCR_VISIBLE_DEVICES=5 python train_arctic.py --keyword Sbw50_fold4 --use_wb --data_fold 4 --boundary_weight 50 &
#ROCR_VISIBLE_DEVICES=6 python train_arctic.py --keyword Sbw50_fold5 --use_wb --data_fold 5 --boundary_weight 50 &
#wait

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword bw10_tw10_fold4 --use_wb --data_fold 4 --boundary_weight 10 --target_weight 10 &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword bw10_tw10_fold5 --use_wb --data_fold 5 --boundary_weight 10 --target_weight 10 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --keyword bw5_tw10_fold1 --use_wb --data_fold 1 --boundary_weight 5 --target_weight 10 &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --keyword bw5_tw10_fold2 --use_wb --data_fold 2 --boundary_weight 5 --target_weight 10 &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --keyword bw5_tw10_fold3 --use_wb --data_fold 3 --boundary_weight 5 --target_weight 10 &
#ROCR_VISIBLE_DEVICES=5 python train_arctic.py --keyword bw5_tw10_fold4 --use_wb --data_fold 4 --boundary_weight 5 --target_weight 10 &
#ROCR_VISIBLE_DEVICES=6 python train_arctic.py --keyword bw5_tw10_fold5 --use_wb --data_fold 5 --boundary_weight 5 --target_weight 10 &
#wait

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword bw5_tw5_fold1 --use_wb --data_fold 1 --boundary_weight 5 --target_weight 5 &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword bw5_tw5_fold2 --use_wb --data_fold 2 --boundary_weight 5 --target_weight 5 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --keyword bw5_tw5_fold3 --use_wb --data_fold 3 --boundary_weight 5 --target_weight 5 &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --keyword bw5_tw5_fold4 --use_wb --data_fold 4 --boundary_weight 5 --target_weight 5 &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --keyword bw5_tw5_fold5 --use_wb --data_fold 5 --boundary_weight 5 --target_weight 5 &
#ROCR_VISIBLE_DEVICES=5 python train_arctic.py --keyword bw10_tw10_fold1 --use_wb --data_fold 1 --boundary_weight 10 --target_weight 10 &
#ROCR_VISIBLE_DEVICES=6 python train_arctic.py --keyword bw10_tw10_fold2 --use_wb --data_fold 2 --boundary_weight 10 --target_weight 10 &
#ROCR_VISIBLE_DEVICES=7 python train_arctic.py --keyword bw10_tw10_fold3 --use_wb --data_fold 3 --boundary_weight 10 --target_weight 10 &
#wait

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword bw1_fold1 --use_wb --data_fold 1 --boundary_weight 1 &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword bw1_fold2 --use_wb --data_fold 2 --boundary_weight 1 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --keyword bw1_fold3 --use_wb --data_fold 3 --boundary_weight 1 &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --keyword bw1_fold4 --use_wb --data_fold 4 --boundary_weight 1 &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --keyword bw1_fold5 --use_wb --data_fold 5 --boundary_weight 1 &
#ROCR_VISIBLE_DEVICES=5 python train_arctic.py --keyword bw5_fold1 --use_wb --data_fold 1 --boundary_weight 5 &
#ROCR_VISIBLE_DEVICES=6 python train_arctic.py --keyword bw5_fold2 --use_wb --data_fold 2 --boundary_weight 5 &
#ROCR_VISIBLE_DEVICES=7 python train_arctic.py --keyword bw5_fold3 --use_wb --data_fold 3 --boundary_weight 5 &
#wait

#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword bw5_fold4 --use_wb --data_fold 4 --boundary_weight 5 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --keyword bw5_fold5 --use_wb --data_fold 5 --boundary_weight 5 &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --keyword bw10_fold1 --use_wb --data_fold 1 --boundary_weight 10 &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --keyword bw10_fold2 --use_wb --data_fold 2 --boundary_weight 10 &
#ROCR_VISIBLE_DEVICES=5 python train_arctic.py --keyword bw10_fold3 --use_wb --data_fold 3 --boundary_weight 10 &
#ROCR_VISIBLE_DEVICES=6 python train_arctic.py --keyword bw10_fold4 --use_wb --data_fold 4 --boundary_weight 10 &
#ROCR_VISIBLE_DEVICES=7 python train_arctic.py --keyword bw10_fold5 --use_wb --data_fold 5 --boundary_weight 10 &
#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword bw15_fold1 --use_wb --data_fold 1 --boundary_weight 15 &
#wait

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword bw15_fold2 --use_wb --data_fold 2 --boundary_weight 15 &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword bw15_fold3 --use_wb --data_fold 3 --boundary_weight 15 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --keyword bw15_fold4 --use_wb --data_fold 4 --boundary_weight 15 &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --keyword bw15_fold5 --use_wb --data_fold 5 --boundary_weight 15 &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --keyword tw5_fold1 --use_wb --data_fold 1 --target_weight 5 &
#ROCR_VISIBLE_DEVICES=5 python train_arctic.py --keyword tw5_fold2 --use_wb --data_fold 2 --target_weight 5 &
#ROCR_VISIBLE_DEVICES=6 python train_arctic.py --keyword tw5_fold3 --use_wb --data_fold 3 --target_weight 5 &
#ROCR_VISIBLE_DEVICES=7 python train_arctic.py --keyword tw5_fold4 --use_wb --data_fold 4 --target_weight 5 &
#wait

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword tw5_fold5 --use_wb --data_fold 5 --target_weight 5 &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword tw10_fold1 --use_wb --data_fold 1 --target_weight 10 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --keyword tw10_fold2 --use_wb --data_fold 2 --target_weight 10 &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --keyword tw10_fold3 --use_wb --data_fold 3 --target_weight 10 &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --keyword tw10_fold4 --use_wb --data_fold 4 --target_weight 10 &
#ROCR_VISIBLE_DEVICES=5 python train_arctic.py --keyword tw10_fold5 --use_wb --data_fold 5 --target_weight 10 &
#ROCR_VISIBLE_DEVICES=6 python train_arctic.py --keyword tw15_fold1 --use_wb --data_fold 1 --target_weight 15 &
#ROCR_VISIBLE_DEVICES=7 python train_arctic.py --keyword tw15_fold2 --use_wb --data_fold 2 --target_weight 15 &
#wait

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword tw15_fold3 --use_wb --data_fold 3 --target_weight 15 &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword tw15_fold4 --use_wb --data_fold 4 --target_weight 15 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --keyword tw15_fold5 --use_wb --data_fold 5 --target_weight 15 &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --keyword tw20_fold1 --use_wb --data_fold 1 --target_weight 20 &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --keyword tw20_fold2 --use_wb --data_fold 2 --target_weight 20 &
#ROCR_VISIBLE_DEVICES=5 python train_arctic.py --keyword tw20_fold3 --use_wb --data_fold 3 --target_weight 20 &
#ROCR_VISIBLE_DEVICES=6 python train_arctic.py --keyword tw20_fold4 --use_wb --data_fold 4 --target_weight 20 &
#ROCR_VISIBLE_DEVICES=7 python train_arctic.py --keyword tw20_fold5 --use_wb --data_fold 5 --target_weight 20 &
#wait

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword ls1_fold1 --stretch_setting 1 --use_wb --data_fold 1 &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword ls1_fold2 --stretch_setting 1 --use_wb --data_fold 2 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --keyword ls1_fold3 --stretch_setting 1 --use_wb --data_fold 3 &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --keyword ls1_fold4 --stretch_setting 1 --use_wb --data_fold 4 &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --keyword ls1_fold5 --stretch_setting 1 --use_wb --data_fold 5 &
#ROCR_VISIBLE_DEVICES=5 python train_arctic.py --keyword ls2_fold1 --stretch_setting 2 --use_wb --data_fold 1 &
#ROCR_VISIBLE_DEVICES=6 python train_arctic.py --keyword ls2_fold2 --stretch_setting 2 --use_wb --data_fold 2 &
#ROCR_VISIBLE_DEVICES=7 python train_arctic.py --keyword ls2_fold3 --stretch_setting 2 --use_wb --data_fold 3 &
#wait

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword l1e1 --use_wb --learning_rate 0.01 &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword l2e5 --use_wb --learning_rate 0.005 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --keyword l2e1 --use_wb --learning_rate 0.001 &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --keyword l3e5 --use_wb --learning_rate 0.0005 &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --keyword l3e1 --use_wb --learning_rate 0.0001 &
#ROCR_VISIBLE_DEVICES=5 python train_arctic.py --keyword l4e4 --use_wb --learning_rate 0.00005 &
#ROCR_VISIBLE_DEVICES=6 python train_arctic.py --keyword l4e1 --use_wb --learning_rate 0.00001 &
#ROCR_VISIBLE_DEVICES=7 python train_arctic.py --keyword l5e5 --use_wb --learning_rate 0.000005 &
#wait