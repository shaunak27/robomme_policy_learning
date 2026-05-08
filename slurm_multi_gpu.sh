#!/bin/bash
#SBATCH -Jmulti_gpu_reserve
#SBATCH --output=logs/multi_gpu_%j.out
#SBATCH --error=logs/multi_gpu_%j.err
#SBATCH --partition="kira-lab"
#SBATCH --account="kira-lab"
#SBATCH --gpus-per-node="a40:8"
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --cpus-per-gpu=3
#SBATCH --qos="short"
#SBATCH --exclude="conroy"

mkdir -p logs

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

nvidia-smi

sleep 86400
