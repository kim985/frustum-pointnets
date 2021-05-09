#!/bin/sh
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH -o /mnt/nfs/scratch1/kfaria/slurm-output/slurm-%j.out

export XDG_RUNTIME_DIR=""
conda activate default
which python
module list

python point_fusion_model.py