#!/bin/bash
#SBATCH -Jtrain_baseline
#SBATCH --output=logs/train_baseline_%j.out
#SBATCH --error=logs/train_baseline_%j.err
#SBATCH --partition="kira-lab"
#SBATCH --account="kira-lab"
#SBATCH --gpus-per-node="a40:8"
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --cpus-per-gpu=12
#SBATCH --qos="long"
#SBATCH --exclude="conroy"

# ---------- Configuration ----------
DATASET="data/robomme_preprocessed_data"
HISTORY_CFG="perceptual-framesamp-modul.yaml"
EXP_NAME="perceptual-framesamp-modul-retrain"
BATCH_SIZE=256
FSDP_DEVICES=8
# -----------------------------------

mkdir -p logs

cat <<EOF > "logs/train_baseline_config_$(date +%Y%m%d_%H%M%S).log"
SLURM_JOB_ID:              ${SLURM_JOB_ID}
SLURM_JOB_NAME:            ${SLURM_JOB_NAME}
LAUNCHED_AT:               $(date)
HOSTNAME:                  $(hostname)

--- Baseline Frame Sampling Training ---
--dataset_path             $DATASET
--model.history_config     $HISTORY_CFG
--exp_name                 $EXP_NAME
--batch_size               $BATCH_SIZE
--fsdp_devices             $FSDP_DEVICES
EOF

srun -u /coc/testnvme/shalbe3/miniconda/envs/robomme/bin/python \
    scripts/train.py mme_vla_suite \
    --skip-tentative \
    --dataset_path "$DATASET" \
    --model.history_config "$HISTORY_CFG" \
    --exp_name "$EXP_NAME" \
    --batch_size $BATCH_SIZE \
    --fsdp_devices $FSDP_DEVICES \
    --num_train_steps 20000 \
    --save_interval 2500 \
    --keep_period 2500 \
    --wandb_enabled \
    --overwrite
