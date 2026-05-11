#!/bin/bash
#SBATCH -Jtrain_qkfs_split
#SBATCH --output=logs/train_qkfs_split_%j.out
#SBATCH --error=logs/train_qkfs_split_%j.err
#SBATCH --partition="kira-lab"
#SBATCH --account="kira-lab"
#SBATCH --gpus-per-node="a40:8"
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --cpus-per-gpu=12
#SBATCH --qos="long"
#SBATCH --exclude="conroy"

# Train QKFS selector on 80/20 split (episodes 0-79 train, 80-99 eval)
# Usage:
#   sbatch slurm_train_qkfs_split.sh

mkdir -p logs

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Training on episodes 0-79 (80%), eval on 80-99 (20%)"
nvidia-smi

srun -u /coc/testnvme/shalbe3/miniconda/envs/robomme/bin/python \
    scripts/train_qkfs.py \
    --dataset_path data/robomme_preprocessed_data \
    --topreward_dir data/topreward_full \
    --exp_name qkfs_split80_v2 \
    --episode_range 0-79 \
    --batch_size 512 \
    --lr 2e-4 \
    --num_train_steps 10000 \
    --warmup_steps 400 \
    --lambda_frame 1.0 \
    --sigma 2.0 \
    --hidden_dim 256 \
    --num_heads 4 \
    --num_query_layers 2 \
    --num_selector_layers 2 \
    --num_recent_frames 4 \
    --max_candidates 512 \
    --wandb_enabled \
    --wandb_project qkfs-selector \
    --log_interval 50 \
    --save_interval 2000 \
    --num_workers 8
