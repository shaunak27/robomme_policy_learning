#!/bin/bash
#SBATCH -Jeval_qkfs
#SBATCH --output=logs/eval_qkfs_%j.out
#SBATCH --error=logs/eval_qkfs_%j.err
#SBATCH --partition="kira-lab"
#SBATCH --account="kira-lab"
#SBATCH --gpus-per-node="a40:1"
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos="short"
#SBATCH --exclude="conroy"

# Evaluate QKFS selector (single GPU, inference only)
# Usage:
#   sbatch slurm_eval_qkfs.sh                                              # latest checkpoint, no-demo tasks
#   sbatch slurm_eval_qkfs.sh runs/ckpts/qkfs/step_5000                   # specific checkpoint
#   sbatch slurm_eval_qkfs.sh runs/ckpts/qkfs/step_final 20 10            # 20 eps, 10 points each
#   sbatch slurm_eval_qkfs.sh runs/ckpts/qkfs/step_final 10 5 all         # all 16 tasks
#   sbatch slurm_eval_qkfs.sh runs/ckpts/qkfs/step_final 20 5 all 80-99   # held-out split eval

mkdir -p logs

CHECKPOINT="${1:-runs/ckpts/qkfs/step_final}"
EPISODES_PER_TASK="${2:-10}"
EVAL_POINTS="${3:-5}"
TASK_SCOPE="${4:-nodemo}"  # "all" for all 16 tasks, default: no-demo only
EPISODE_RANGE="${5:-}"     # e.g. "80-99" for held-out eval split

# Derive output dir from checkpoint name
CKPT_NAME=$(basename "$CHECKPOINT")
OUTPUT_DIR="runs/eval_qkfs/${CKPT_NAME}"

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Episodes/task: $EPISODES_PER_TASK, eval points/episode: $EVAL_POINTS"
nvidia-smi

EXTRA_ARGS=""
if [ "$TASK_SCOPE" = "all" ]; then
    EXTRA_ARGS="--all_tasks"
    echo "Running on ALL 16 tasks"
fi
if [ -n "$EPISODE_RANGE" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --episode_range $EPISODE_RANGE"
    echo "Episode range: $EPISODE_RANGE"
fi

srun -u /coc/testnvme/shalbe3/miniconda/envs/robomme/bin/python \
    scripts/eval_qkfs.py \
    --checkpoint "$CHECKPOINT" \
    --dataset_path data/robomme_preprocessed_data \
    --episodes_per_task "$EPISODES_PER_TASK" \
    --eval_points_per_episode "$EVAL_POINTS" \
    --output_dir "$OUTPUT_DIR" \
    $EXTRA_ARGS
