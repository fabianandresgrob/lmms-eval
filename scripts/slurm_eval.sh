#!/bin/bash
#SBATCH --job-name=lmms-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#
# GPU, CPU, mem, and time are set dynamically by submit_all.sh via --gres, --cpus-per-task, --mem, --time flags.

set -e

# ---- Validate required env vars ----
for var in MODEL_TYPE PRETRAINED MODEL_NAME; do
    if [ -z "${!var}" ]; then
        echo "ERROR: \$$var is not set."
        exit 1
    fi
done

if [ -z "$SCRATCH" ]; then
    echo "ERROR: \$SCRATCH is not set."
    exit 1
fi

TASKS="${TASKS:-vlms_are_biased,vilp,vlind_bench}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LMMS_EVAL_DIR="$HOME/projects/lmms-eval"
RESULTS_DIR="$SCRATCH/results/lmms-eval"

# ---- Activate environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lmms-eval

# ---- CUDA setup ----
TORCH_CUDA_VER=$(python -c "import torch; print(torch.version.cuda)")
if [ -d "/usr/local/cuda-$TORCH_CUDA_VER" ]; then
    export CUDA_HOME="/usr/local/cuda-$TORCH_CUDA_VER"
else
    export CUDA_HOME="/usr/local/cuda"
fi
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ---- Determine model_args ----
NUM_GPUS=$(nvidia-smi -L | wc -l)
MODEL_ARGS="pretrained=$PRETRAINED"
if [ "$NUM_GPUS" -gt 1 ]; then
    MODEL_ARGS="${MODEL_ARGS},device_map=auto"
    echo "$(date): Multi-GPU mode — $NUM_GPUS GPUs, device_map=auto"
fi

# ---- Run evaluation ----
echo "$(date): Starting evaluation"
echo "  Model type:  $MODEL_TYPE"
echo "  Pretrained:  $PRETRAINED"
echo "  Model name:  $MODEL_NAME"
echo "  Tasks:       $TASKS"
echo "  Batch size:  $BATCH_SIZE"
echo "  GPUs:        $NUM_GPUS"
echo "  Results dir: $RESULTS_DIR/$MODEL_NAME"

mkdir -p "$RESULTS_DIR" logs

cd "$LMMS_EVAL_DIR"
python -m lmms_eval \
    --model "$MODEL_TYPE" \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$RESULTS_DIR/$MODEL_NAME" \
    --log_samples \
    --verbosity INFO

echo "$(date): Evaluation complete. Results in $RESULTS_DIR/$MODEL_NAME"
