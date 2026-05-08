#!/bin/bash
#SBATCH -Jeval_uni_patternlock
#SBATCH --output=logs/eval_uniform_patternlock_%j.out
#SBATCH --error=logs/eval_uniform_patternlock_%j.err
#SBATCH --partition="kira-lab"
#SBATCH --account="kira-lab"
#SBATCH --gpus-per-node="a40:2"
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=12
#SBATCH --qos="long"
#SBATCH --exclude="conroy"

# ---------- Configuration ----------
TASK="PatternLock"
METHOD="uniform"
MODEL_TYPE="perceptual-framesamp-modul"
CONFIG_TYPE="mme_vla_suite"
CKPT_ID=79999
SEED=42
GPU_SERVER=0
GPU_CLIENT=1
# -----------------------------------

mkdir -p logs

# Find a free port
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Using port $PORT for task=$TASK method=$METHOD"

cat <<EOF > "logs/eval_config_${METHOD}_${TASK}_$(date +%Y%m%d_%H%M%S).log"
SLURM_JOB_ID:    ${SLURM_JOB_ID}
HOSTNAME:         $(hostname)
TASK:             $TASK
METHOD:           $METHOD
MODEL_TYPE:       $MODEL_TYPE
CKPT_ID:          $CKPT_ID
SEED:             $SEED
PORT:             $PORT
EOF

# ---- Launch policy server on GPU_SERVER ----
echo "Starting policy server on GPU $GPU_SERVER ..."
CUDA_VISIBLE_DEVICES=$GPU_SERVER /coc/testnvme/shalbe3/miniconda/envs/robomme/bin/uv run scripts/serve_policy.py \
    --seed=$SEED \
    --port=$PORT \
    policy:checkpoint \
    --policy.dir=runs/ckpts/$CONFIG_TYPE/$MODEL_TYPE/$CKPT_ID \
    --policy.config=$CONFIG_TYPE &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start ..."
for i in $(seq 1 120); do
    if python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost',$PORT)); s.close()" 2>/dev/null; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

# ---- Launch eval client on GPU_CLIENT ----
echo "Starting eval client on GPU $GPU_CLIENT for task=$TASK ..."
CUDA_VISIBLE_DEVICES=$GPU_CLIENT /coc/testnvme/shalbe3/micromamba/envs/robomme/bin/python \
    examples/robomme/eval.py \
    --args.model-seed=$SEED \
    --args.port=$PORT \
    --args.policy-name="${METHOD}_${TASK}" \
    --args.model-ckpt-id=$CKPT_ID \
    --args.only-tasks=$TASK

eval_exit=$?
echo "Eval finished with exit code $eval_exit"

# Cleanup server
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
exit $eval_exit
