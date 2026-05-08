#!/bin/bash
#SBATCH -Jbuild_robomme_dataset
#SBATCH --output=logs/build_dataset_%j.out
#SBATCH --error=logs/build_dataset_%j.err
#SBATCH --partition="kira-lab"
#SBATCH --account="kira-lab"
#SBATCH --gpus-per-node="a40:1"
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-gpu=12
#SBATCH --qos="long"
#SBATCH --exclude="conroy"

# ---------- Configuration ----------
RAW_DATA_PATH="data/robomme_data_h5"
PREPROCESSED_DATA_PATH="data/robomme_preprocessed_data"
# -----------------------------------

mkdir -p logs

cat <<EOF > "logs/build_dataset_config_$(date +%Y%m%d_%H%M%S).log"
SLURM_JOB_ID:              ${SLURM_JOB_ID}
SLURM_JOB_NAME:            ${SLURM_JOB_NAME}
LAUNCHED_AT:               $(date)
HOSTNAME:                  $(hostname)

--- Build Dataset Configuration ---
--raw_data_path            $RAW_DATA_PATH
--preprocessed_data_path   $PREPROCESSED_DATA_PATH
EOF

srun -u /coc/testnvme/shalbe3/miniconda/envs/robomme/bin/python \
    scripts/build_dataset.py \
    --dataset_type robomme_pkl \
    --raw_data_path "$RAW_DATA_PATH" \
    --preprocessed_data_path "$PREPROCESSED_DATA_PATH"
