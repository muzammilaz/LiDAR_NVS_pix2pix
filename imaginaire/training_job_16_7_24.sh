#!/bin/bash -l
#SBATCH --job-name=unit_imaginaire_30_7_24
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --output=/home/woody/i9vl/i9vl106h/Results/unit_imaginaire_30_7_24/msg_rtx2080ti.out
#SBATCH --error=/home/woody/i9vl/i9vl106h/Results/unit_imaginaire_30_7_24/msg_rtx2080ti.err
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
log=/home/woody/i9vl/i9vl106h/Results/unit_imaginaire_30_7_24/experiment.log

export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80

# log current git commit
if ! command -v git &> /dev/null
then
    echo git not found - trying to load git module...
    module load git
fi
git log -1 | head -1 > $log

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/projects/pix2pixhd/kitti360/16_7_24_test.yaml --single_gpu --wandb_id muzammil --wandb_name imaginaire_test --checkpoint logs/2024_0730_1934_19_16_7_24_test/epoch_00251_iteration_000280000_checkpoint.pt --output_dir logs/UniT_test_06_08_2024