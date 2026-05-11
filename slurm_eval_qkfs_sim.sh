#!/bin/bash
#SBATCH -Jeval_qkfs_sim
#SBATCH --output=logs/eval_qkfs_sim_%A_%a.out
#SBATCH --error=logs/eval_qkfs_sim_%A_%a.err
#SBATCH --partition="kira-lab"
#SBATCH --account="kira-lab"
#SBATCH --gpus-per-node="a40:2"
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=12
#SBATCH --qos="long"
#SBATCH --exclude="conroy"
#SBATCH --array=0-7

# Evaluate density VLA with QKFS frame selector in simulation.
# Array job: 8 tasks, each handles 2 of the 16 tasks.
#
# Usage:
#   sbatch slurm_eval_qkfs_sim.sh                          # all 8 array jobs
#   sbatch --array=0 slurm_eval_qkfs_sim.sh                # just first pair
#   sbatch slurm_eval_qkfs_sim.sh path/to/qkfs/ckpt        # custom QKFS ckpt

QKFS_CKPT="${1:-runs/ckpts/qkfs/qkfs_split80/step_final}"

# ---------- Configuration ----------
METHOD="qkfs_sim"
BASE_MODEL_TYPE="perceptual-density-modul"
CONFIG_TYPE="mme_vla_suite"
CKPT_ID=17500
SEED=7
GPU_SERVER=0
GPU_CLIENT=1
# -----------------------------------

# 16 tasks split into 8 groups of 2
TASK_GROUPS=(
    "BinFill,StopCube"
    "PickXtimes,SwingXtimes"
    "ButtonUnmask,VideoUnmask"
    "VideoUnmaskSwap,ButtonUnmaskSwap"
    "PickHighlight,VideoRepick"
    "VideoPlaceButton,VideoPlaceOrder"
    "MoveCube,InsertPeg"
    "PatternLock,RouteStick"
)

TASKS="${TASK_GROUPS[$SLURM_ARRAY_TASK_ID]}"
echo "Array task $SLURM_ARRAY_TASK_ID: tasks=$TASKS"

mkdir -p logs

VLA_CKPT_DIR="runs/ckpts/$CONFIG_TYPE/$BASE_MODEL_TYPE"

# Find a free port
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Using port $PORT for tasks=$TASKS method=$METHOD"

cat <<EOF > "logs/eval_config_${METHOD}_${TASKS//,/_}_$(date +%Y%m%d_%H%M%S).log"
SLURM_JOB_ID:    ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
HOSTNAME:         $(hostname)
TASKS:            $TASKS
METHOD:           $METHOD
QKFS_CKPT:       $QKFS_CKPT
BASE_MODEL:       $BASE_MODEL_TYPE
CKPT_ID:          $CKPT_ID
SEED:             $SEED
PORT:             $PORT
EOF

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Tasks: $TASKS"
echo "QKFS checkpoint: $QKFS_CKPT"
echo "VLA checkpoint: $BASE_MODEL_TYPE/$CKPT_ID"
nvidia-smi

# ---- Launch policy server on GPU_SERVER ----
echo "Starting policy server on GPU $GPU_SERVER with QKFS selector ..."
CUDA_VISIBLE_DEVICES=$GPU_SERVER /coc/testnvme/shalbe3/miniconda/envs/robomme/bin/uv run scripts/serve_policy.py \
    --seed=$SEED \
    --port=$PORT \
    --qkfs-checkpoint=$QKFS_CKPT \
    policy:checkpoint \
    --policy.dir=$VLA_CKPT_DIR/$CKPT_ID \
    --policy.config=$CONFIG_TYPE &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start ..."
for i in $(seq 1 180); do
    if python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost',$PORT)); s.close()" 2>/dev/null; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

# ---- Launch eval client on GPU_CLIENT ----
echo "Starting eval client on GPU $GPU_CLIENT for tasks=$TASKS ..."
EVAL_ARGS=(
    --args.model-seed=$SEED
    --args.port=$PORT
    --args.policy-name="${METHOD}_${TASKS//,/_}"
    --args.model-ckpt-id=$CKPT_ID
    --args.only-tasks=$TASKS
)

CUDA_VISIBLE_DEVICES=$GPU_CLIENT /coc/testnvme/shalbe3/micromamba/envs/robomme/bin/python \
    examples/robomme/eval.py "${EVAL_ARGS[@]}"

eval_exit=$?
echo "Eval finished with exit code $eval_exit"

# Cleanup server
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
exit $eval_exit
