#!/bin/bash
#SBATCH -Jconsolidate_emb
#SBATCH --output=logs/consolidate_emb_%j.out
#SBATCH --error=logs/consolidate_emb_%j.err
#SBATCH --partition="kira-lab"
#SBATCH --account="kira-lab"
#SBATCH --gpus-per-node="a40:1"
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-gpu=12
#SBATCH --qos="long"
#SBATCH --exclude="conroy"

# ---------- Configuration ----------
DATASET_PATH="data/robomme_preprocessed_data"
WORKERS=16
# -----------------------------------

mkdir -p logs

cat <<EOF > "logs/consolidate_config_$(date +%Y%m%d_%H%M%S).log"
SLURM_JOB_ID:              ${SLURM_JOB_ID}
SLURM_JOB_NAME:            ${SLURM_JOB_NAME}
LAUNCHED_AT:               $(date)
HOSTNAME:                  $(hostname)

--- Consolidate Embeddings ---
--dataset_path             $DATASET_PATH
--workers                  $WORKERS
EOF

srun -u /coc/testnvme/shalbe3/miniconda/envs/robomme/bin/python \
    scripts/consolidate_embeddings.py \
    --dataset_path "$DATASET_PATH" \
    --workers $WORKERS
