#!/bin/bash
#SBATCH -Jtopreward_resume
#SBATCH --output=logs/topreward_resume_%A_%a.out
#SBATCH --error=logs/topreward_resume_%A_%a.err
#SBATCH --partition="kira-lab,overcap"
#SBATCH --account="kira-lab"
#SBATCH --gpus-per-node="a40:1"
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-gpu=12
#SBATCH --qos="debug"
#SBATCH --exclude="conroy"
#SBATCH --array=0-2

# ---------- Incomplete tasks ----------
TASKS=(BinFill PickXtimes SwingXtimes)
# Episodes that still need processing (start from where it crashed)
START_EPS=(82 84 78)

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
START_EP=${START_EPS[$SLURM_ARRAY_TASK_ID]}

# ---------- Configuration ----------
RAW_DATA="data/robomme_data_h5"
OUTPUT_DIR="data/topreward_full"
NUM_PREFIXES=8
TARGET_FPS=4.0
MODEL="Qwen/Qwen3-VL-8B-Instruct"
# -----------------------------------

mkdir -p logs

echo "===== TOPReward Resume ====="
echo "SLURM_JOB_ID:       ${SLURM_JOBID}"
echo "TASK:                ${TASK}"
echo "START_EP:            ${START_EP}"
echo "HOSTNAME:            $(hostname)"
echo "GPU:                 $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "STARTED:             $(date)"
echo "=============================="

# Process each missing episode individually
for EP in $(seq $START_EP 99); do
    echo "--- Processing ${TASK} episode ${EP} ---"
    srun -u /coc/testnvme/shalbe3/miniconda/envs/robomme/bin/python \
        scripts/pseudo_label_topreward.py \
        --raw_data_path "$RAW_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --task "$TASK" \
        --episode $EP \
        --num_prefixes $NUM_PREFIXES \
        --target_fps $TARGET_FPS \
        --model_name "$MODEL"
done
