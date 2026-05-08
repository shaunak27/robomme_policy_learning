#!/bin/bash
#SBATCH -Jtrain_selector_v2
#SBATCH --output=logs/train_selector_v2_%j.out
#SBATCH --error=logs/train_selector_v2_%j.err
#SBATCH --partition="kira-lab"
#SBATCH --account="kira-lab"
#SBATCH --gpus-per-node="a40:8"
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --cpus-per-gpu=12
#SBATCH --qos="long"
#SBATCH --exclude="conroy"

# ---------- Configuration ----------
VLA_CKPT="runs/ckpts/mme_vla_suite/perceptual-framesamp-modul/79999/params"
DATASET="data/robomme_preprocessed_data"
EXP_NAME="rl_selector_v2"
BATCH_SIZE=64
# -----------------------------------

mkdir -p logs

cat <<EOF > "logs/train_selector_v2_config_$(date +%Y%m%d_%H%M%S).log"
SLURM_JOB_ID:              ${SLURM_JOB_ID}
SLURM_JOB_NAME:            ${SLURM_JOB_NAME}
LAUNCHED_AT:               $(date)
HOSTNAME:                  $(hostname)

--- Train Selector V2 (Fused) ---
--vla_checkpoint_path      $VLA_CKPT
--dataset_path             $DATASET
--exp_name                 $EXP_NAME
--batch_size               $BATCH_SIZE
EOF

srun -u /coc/testnvme/shalbe3/miniconda/envs/robomme/bin/python \
    scripts/train_selector_v2.py \
    --vla_checkpoint_path "$VLA_CKPT" \
    --dataset_path "$DATASET" \
    --exp_name "$EXP_NAME" \
    --batch_size $BATCH_SIZE \
    --wandb_enabled \
    --tasks PatternLock RouteStick
