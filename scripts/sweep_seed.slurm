#!/bin/bash
#SBATCH --job-name=sweep_seed
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=4:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%N_%A_%a.out

SEED_VALUES=(1 2 3)

seed=${SEED_VALUES[$SLURM_ARRAY_TASK_ID]}

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate plasmos-demo

srun python3 run_wandb.py \
  --n 1000000 \
  --M 100 \
  --dt 0.01 \
  --dv 1 \
  --seed "$seed"
