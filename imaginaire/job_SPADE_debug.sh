#!/bin/bash -l
#SBATCH --job-name=spade_debug
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --output=/home/woody/i9vl/i9vl106h/Results/spade_debug/msg_rtx2080ti.out
#SBATCH --error=/home/woody/i9vl/i9vl106h/Results/spade_debug/msg_rtx2080ti.err
#SBATCH --export=NONE


# do not export environment variables
unset SLURM_EXPORT_ENV

# jobs always start in /home/hpc/i9vl/i9vl106h/imaginaire
cd /home/hpc/i9vl/i9vl106h/imaginaire

# activate virtual environment
source /apps/jupyterhub/jh3.1.1-py3.11/bin/activate /home/woody/i9vl/i9vl106h/software/privat/conda/envs/imaginaire

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

CUDA_VISIBLE_DEVICES=0 python train.py --config /home/woody/i9vl/i9vl106h/configs/projects/spade/kitti/DEBUG_run.yaml --single_gpu --wandb_id muzammil --wandb_name imaginaire_test --logdir /home/woody/i9vl/i9vl106h/logs/spade_debug