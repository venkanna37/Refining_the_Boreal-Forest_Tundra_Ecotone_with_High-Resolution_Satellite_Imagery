#!/bin/bash
#SBATCH --job-name=LS2_Arctic
#SBATCH --account=project_465002698
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=small-g
#SBATCH --time=15:00:00

hostname
rocm-smi
echo $CUDA_VISIBLE_DEVICES

ROCR_VISIBLE_DEVICES=0 python train_arctic.py --keyword ls2_fold4 --stretch_setting 2 --use_wb --data_fold 4 &
ROCR_VISIBLE_DEVICES=1 python train_arctic.py --keyword ls2_fold5 --stretch_setting 2 --use_wb --data_fold 5 &
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