#!/bin/bash
#SBATCH --job-name=sweep_dt
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=4:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%N_%A_%a.out

DT_VALUES=(0.1 0.01 0.001)

dt=${DT_VALUES[$SLURM_ARRAY_TASK_ID]}

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate plasmos-demo

srun python3 run_wandb.py \
  --n 1000000 \
  --M 100 \
  --dt "$dt" \
  --dv 1 \
  --seed 42
