#!/bin/bash -l
#SBATCH --job-name=lidar4d_kitti360_4950_exp_01_07_2024
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/home/woody/i9vl/i9vl106h/Results/lidar4d_kitti360_4950_exp_01_07_2024/msg_rtx2080ti.out
#SBATCH --error=/home/woody/i9vl/i9vl106h/Results/lidar4d_kitti360_4950_exp_01_07_2024/msg_rtx2080ti.err
#SBATCH --export=NONE


# do not export environment variables
unset SLURM_EXPORT_ENV

# jobs always start in /home/hpc/i9vl/i9vl106h
cd /home/hpc/i9vl/i9vl106h/LiDAR4D

# activate virtual environment
source /apps/jupyterhub/jh3.1.1-py3.11/bin/activate /home/woody/i9vl/i9vl106h/software/privat/conda/envs/lidar4d

# load cuda module
module load cuda/11.8.0
# module load cudnn/8.8.0.121-11.8 

# log file
log=/home/woody/i9vl/i9vl106h/Results/lidar4d_kitti360_4950_exp_01_07_2024/experiment.log

# log current git commit
if ! command -v git &> /dev/null
then
    echo git not found - trying to load git module...
    module load git
fi
git log -1 | head -1 > $log

CUDA_VISIBLE_DEVICES=0 python main_lidar4d.py \
--config configs/kitti360_4950.txt \
--workspace log/kitti360_lidar4d_f4950_release_2 \
--lr 1e-2 \
--num_rays_lidar 2048 \
--iters 40000 \
--alpha_d 1 \
--alpha_i 0.1 \
--alpha_r 0.01 \
