#!/bin/bash
#SBATCH --job-name=ArcticH9
#SBATCH --account=project_465002698
#SBATCH --nodes=1
#SBATCH --gpus-per-node=7
#SBATCH --ntasks-per-node=7
#SBATCH --gpus-per-task=1
#SBATCH --mem=0
#SBATCH --partition=standard-g
#SBATCH --output=/scratch/project_465002698/venky/projects/arctic/out_files/%x_%j.out
#SBATCH --error=/scratch/project_465002698/venky/projects/arctic/out_files/%x_%j.err
#SBATCH --time=48:00:00

hostname
rocm-smi
echo $CUDA_VISIBLE_DEVICES

#ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword Sbw10_fold1 --use_wb --data_fold 1 --boundary_weight 10 &
#ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword Sbw10_fold2 --use_wb --data_fold 2 --boundary_weight 10 &
#ROCR_VISIBLE_DEVICES=2 python train_arctic.py --keyword Sbw10_fold3 --use_wb --data_fold 3 --boundary_weight 10 &
#ROCR_VISIBLE_DEVICES=3 python train_arctic.py --keyword Sbw10_fold4 --use_wb --data_fold 4 --boundary_weight 10 &
#ROCR_VISIBLE_DEVICES=4 python train_arctic.py --keyword Sbw10_fold5 --use_wb --data_fold 5 --boundary_weight 10 &
#ROCR_VISIBLE_DEVICES=5 python train_arctic.py --keyword Sbw20_fold1 --use_wb --data_fold 1 --boundary_weight 20 &
#ROCR_VISIBLE_DEVICES=6 python train_arctic.py --keyword Sbw20_fold2 --use_wb --data_fold 2 --boundary_weight 20 &
#ROCR_VISIBLE_DEVICES=7 python train_arctic.py --keyword Sbw20_fold3 --use_wb --data_fold 3 --boundary_weight 20 &
#wait

ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword Sbw20_fold4 --use_wb --data_fold 4 --boundary_weight 20 &
ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword Sbw20_fold5 --use_wb --data_fold 5 --boundary_weight 20 &
ROCR_VISIBLE_DEVICES=2 python train_arctic.py --keyword Sbw50_fold1 --use_wb --data_fold 1 --boundary_weight 50 &
ROCR_VISIBLE_DEVICES=3 python train_arctic.py --keyword Sbw50_fold2 --use_wb --data_fold 2 --boundary_weight 50 &
ROCR_VISIBLE_DEVICES=4 python train_arctic.py --keyword Sbw50_fold3 --use_wb --data_fold 3 --boundary_weight 50 &
ROCR_VISIBLE_DEVICES=5 python train_arctic.py --keyword Sbw50_fold4 --use_wb --data_fold 4 --boundary_weight 50 &
ROCR_VISIBLE_DEVICES=6 python train_arctic.py --keyword Sbw50_fold5 --use_wb --data_fold 5 --boundary_weight 50 &
wait

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

ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword tw15_fold3 --use_wb --data_fold 3 --target_weight 15 &
ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword tw15_fold4 --use_wb --data_fold 4 --target_weight 15 &
ROCR_VISIBLE_DEVICES=2 python train_arctic.py --keyword tw15_fold5 --use_wb --data_fold 5 --target_weight 15 &
ROCR_VISIBLE_DEVICES=3 python train_arctic.py --keyword tw20_fold1 --use_wb --data_fold 1 --target_weight 20 &
ROCR_VISIBLE_DEVICES=4 python train_arctic.py --keyword tw20_fold2 --use_wb --data_fold 2 --target_weight 20 &
ROCR_VISIBLE_DEVICES=5 python train_arctic.py --keyword tw20_fold3 --use_wb --data_fold 3 --target_weight 20 &
ROCR_VISIBLE_DEVICES=6 python train_arctic.py --keyword tw20_fold4 --use_wb --data_fold 4 --target_weight 20 &
ROCR_VISIBLE_DEVICES=7 python train_arctic.py --keyword tw20_fold5 --use_wb --data_fold 5 --target_weight 20 &
wait

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