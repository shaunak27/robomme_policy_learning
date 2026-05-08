#!/bin/bash
#SBATCH -Jbuild_robomme_%a
#SBATCH --output=logs/build_dataset_%A_%a.out
#SBATCH --error=logs/build_dataset_%A_%a.err
#SBATCH --partition="kira-lab"
#SBATCH --account="kira-lab"
#SBATCH --gpus-per-node="a40:1"
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-gpu=12
#SBATCH --qos="long"
#SBATCH --exclude="conroy,perseverance"
#SBATCH --array=14-15

# ---------- Configuration ----------
RAW_DATA_PATH="data/robomme_data_h5"
PER_TASK_DIR="data/robomme_preprocessed_data/per_task"
PYTHON="/coc/testnvme/shalbe3/miniconda/envs/robomme/bin/python"
# -----------------------------------

# Map array index to h5 file
H5_FILES=($(ls ${RAW_DATA_PATH}/record_dataset_*.h5 | sort))
H5_FILE="${H5_FILES[$SLURM_ARRAY_TASK_ID]}"

if [ -z "$H5_FILE" ]; then
    echo "No h5 file for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Extract task name: record_dataset_BinFill.h5 -> BinFill
TASK_NAME=$(basename "$H5_FILE" .h5 | sed 's/record_dataset_//')
OUTPUT_DIR="${PER_TASK_DIR}/${TASK_NAME}"

mkdir -p logs

echo "Job $SLURM_ARRAY_TASK_ID: processing $TASK_NAME"
echo "  h5_file:    $H5_FILE"
echo "  output_dir: $OUTPUT_DIR"
echo "  hostname:   $(hostname)"
echo "  started:    $(date)"

srun -u $PYTHON scripts/build_dataset_single_task.py \
    --h5_file "$H5_FILE" \
    --output_dir "$OUTPUT_DIR"
