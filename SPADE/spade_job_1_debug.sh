#!/bin/bash -l
#SBATCH --job-name=spade_debug
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --output=/home/woody/i9vl/i9vl106h/Results/spade_debug/msg_rtx2080ti.out
#SBATCH --error=/home/woody/i9vl/i9vl106h/Results/spade_debug/msg_rtx2080ti.err
#SBATCH --export=NONE


# do not export environment variables
unset SLURM_EXPORT_ENV

# jobs always start in /home/hpc/i9vl/i9vl106h/imaginaire
cd /home/hpc/i9vl/i9vl106h/SPADE

# activate virtual environment
source /apps/jupyterhub/jh3.1.1-py3.11/bin/activate /home/woody/i9vl/i9vl106h/software/privat/conda/envs/spade

# load cuda module
module load cuda/11.8.0
# module load cudnn/8.8.0.121-11.8 

# log file
log=/home/woody/i9vl/i9vl106h/Results/spade_debug/experiment.log

export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

# log current git commit
if ! command -v git &> /dev/null
then
    echo git not found - trying to load git module...
    module load git
fi
git log -1 | head -1 > $log

CUDA_VISIBLE_DEVICES=0 python -u train.py --name spade_debug --dataset_mode pix2pix --dataroot /home/woody/i9vl/i9vl106h/data/lidar4d_data --no_instance --label_nc 0 --preprocess_mode fixed_wh --load_size 1000 --crop 896 --aspect_ratio 4 --display_freq 1 --niter 20 --batchSize 4 --use_masked_training --mask_in_data --checkpoints_dir /home/woody/i9vl/i9vl106h/logs --output_nc 1