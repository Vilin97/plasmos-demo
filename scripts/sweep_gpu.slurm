#!/bin/bash
#SBATCH --job-name=sweep_gpu
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%N_%A_%a.out

GPU_COUNTS=(1 2 4)

ngpu=${GPU_COUNTS[$SLURM_ARRAY_TASK_ID]}

# Restrict JAX to only see the first ngpu GPUs
DEVICES=$(seq -s, 0 $((ngpu - 1)))
export CUDA_VISIBLE_DEVICES=$DEVICES

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate plasmos-demo

srun python3 run_wandb.py \
  --n 100000000 \
  --M 100 \
  --dt 0.01 \
  --dv 1 \
  --seed 42
