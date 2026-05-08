#!/bin/bash
#SBATCH -Jtopreward_label
#SBATCH --output=logs/topreward_%A_%a.out
#SBATCH --error=logs/topreward_%A_%a.err
#SBATCH --partition="kira-lab"
#SBATCH --account="kira-lab"
#SBATCH --gpus-per-node="a40:1"
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-gpu=12
#SBATCH --qos="short"
#SBATCH --exclude="conroy"
#SBATCH --array=0-15

# ---------- Task list (one per array index) ----------
TASKS=(
    BinFill
    ButtonUnmask
    ButtonUnmaskSwap
    InsertPeg
    MoveCube
    PatternLock
    PickHighlight
    PickXtimes
    RouteStick
    StopCube
    SwingXtimes
    VideoPlaceButton
    VideoPlaceOrder
    VideoRepick
    VideoUnmask
    VideoUnmaskSwap
)

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

# ---------- Configuration ----------
RAW_DATA="data/robomme_data_h5"
OUTPUT_DIR="data/topreward_full"
EPISODES_PER_TASK=100
NUM_PREFIXES=8
TARGET_FPS=4.0
MODEL="Qwen/Qwen3-VL-8B-Instruct"
# -----------------------------------

mkdir -p logs

echo "===== TOPReward Labeling ====="
echo "SLURM_JOB_ID:       ${SLURM_JOBID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "TASK:                ${TASK}"
echo "HOSTNAME:            $(hostname)"
echo "GPU:                 $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "STARTED:             $(date)"
echo "=============================="

srun -u /coc/testnvme/shalbe3/miniconda/envs/robomme/bin/python \
    scripts/pseudo_label_topreward.py \
    --raw_data_path "$RAW_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --task "$TASK" \
    --episodes_per_task $EPISODES_PER_TASK \
    --num_prefixes $NUM_PREFIXES \
    --target_fps $TARGET_FPS \
    --model_name "$MODEL"
