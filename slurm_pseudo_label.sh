#!/bin/bash
#SBATCH -Jpseudo_label_phases
#SBATCH --output=logs/pseudo_label_%j.out
#SBATCH --error=logs/pseudo_label_%j.err
#SBATCH --partition="kira-lab"
#SBATCH --account="kira-lab"
#SBATCH --gpus-per-node="a40:1"
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-gpu=12
#SBATCH --qos="long"
#SBATCH --exclude="conroy"

# ---------- Configuration ----------
RAW_DATA="data/robomme_data_h5"
OUTPUT_DIR="data/pseudo_labels"
FRACTION=0.10
EPISODES_PER_TASK=3
CLIP_MODEL="openai/clip-vit-large-patch14"
GEMINI_MODEL="gemini-2.5-flash-lite"
USE_TRIPLES="--use_triples"     # Use image triples (faster than video upload)
# SKIP_MLLM="--skip_mllm"       # Uncomment for CLIP-only dry run
SKIP_MLLM=""
SMOOTH_KERNEL=5
# MAX_SEGMENTS="--max_segments 5" # Uncomment for debugging
MAX_SEGMENTS=""
# -----------------------------------

mkdir -p logs

cat <<EOF > "logs/pseudo_label_config_$(date +%Y%m%d_%H%M%S).log"
SLURM_JOB_ID:              ${SLURM_JOB_ID}
SLURM_JOB_NAME:            ${SLURM_JOB_NAME}
LAUNCHED_AT:               $(date)
HOSTNAME:                  $(hostname)

--- Pseudo-Label Configuration ---
--raw_data_path            $RAW_DATA
--output_dir               $OUTPUT_DIR
--fraction                 $FRACTION
--clip_model               $CLIP_MODEL
--gemini_model             $GEMINI_MODEL
--smooth_kernel            $SMOOTH_KERNEL
USE_TRIPLES:               $USE_TRIPLES
SKIP_MLLM:                 $SKIP_MLLM
MAX_SEGMENTS:              $MAX_SEGMENTS
EOF

srun -u /coc/testnvme/shalbe3/miniconda/envs/robomme/bin/python \
    scripts/pseudo_label_phases.py \
    --raw_data_path "$RAW_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --fraction $FRACTION \
    --episodes_per_task $EPISODES_PER_TASK \
    --clip_model "$CLIP_MODEL" \
    --gemini_model "$GEMINI_MODEL" \
    --smooth_kernel $SMOOTH_KERNEL \
    $USE_TRIPLES \
    $SKIP_MLLM \
    $MAX_SEGMENTS
