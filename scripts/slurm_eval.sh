#!/bin/bash
#SBATCH --job-name=lmms-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# partition, qos, GPU, CPU, mem, and time are all set dynamically by submit_all.sh.
# Defaults below are only used when running slurm_eval.sh directly without submit_all.sh.
#SBATCH --partition=mcml-hgx-a100-80x4,mcml-hgx-h100-94x4
#SBATCH --qos=mcml

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

# Honor externally provided TASKS from submission; fall back only if unset.
# submit_all.sh encodes commas as ';' because sbatch --export is comma-delimited.
REQUESTED_TASKS_RAW="${TASKS:-vlms_are_biased,vilp,vlind_bench,vlind_bench_oe}"
REQUESTED_TASKS="${REQUESTED_TASKS_RAW//;/,}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LMMS_EVAL_DIR="$HOME/projects/lmms-eval"
RESULTS_DIR="$SCRATCH/results/lmms-eval"

# Return success if ALL requested tasks already have at least one task-specific
# samples file for the given model directory.
#
# lmms-eval writes outputs as:
#   $RESULTS_DIR/$MODEL_NAME/<hf_model_slug>/<timestamp>_results.json
#   $RESULTS_DIR/$MODEL_NAME/<hf_model_slug>/<timestamp>_samples_<task>.jsonl
# (i.e., no per-task subdirectory under $MODEL_NAME)
all_tasks_complete() {
    local model_results_dir="$1"
    local tasks_csv="$2"
    local task

    IFS=',' read -r -a task_list <<< "$tasks_csv"
    for task in "${task_list[@]}"; do
        # Trim surrounding whitespace
        task="${task#${task%%[![:space:]]*}}"
        task="${task%${task##*[![:space:]]}}"

        if ! compgen -G "$model_results_dir"/*/*_samples_"$task".jsonl > /dev/null 2>&1; then
            return 1
        fi
    done

    return 0
}

task_has_output() {
    local model_results_dir="$1"
    local task="$2"

    compgen -G "$model_results_dir"/*/*_samples_"$task".jsonl > /dev/null 2>&1
}

pending_tasks_csv() {
    local model_results_dir="$1"
    local tasks_csv="$2"
    local task
    local pending=""

    IFS=',' read -r -a task_list <<< "$tasks_csv"
    for task in "${task_list[@]}"; do
        # Trim surrounding whitespace
        task="${task#${task%%[![:space:]]*}}"
        task="${task%${task##*[![:space:]]}}"

        if [ -z "$task" ]; then
            continue
        fi

        if ! task_has_output "$model_results_dir" "$task"; then
            if [ -z "$pending" ]; then
                pending="$task"
            else
                pending="$pending,$task"
            fi
        fi
    done

    echo "$pending"
}

# Override lmms-eval's remote-fs detection — keep datasets cache on $SCRATCH,
# not redirected to /tmp (which is small on compute nodes).
export LMMS_EVAL_DATASETS_CACHE="$SCRATCH/.cache/huggingface/datasets"

# ---- Activate environment ----
UV_VENV_DIR="$LMMS_EVAL_DIR/.venv"
if [ -f "$UV_VENV_DIR/bin/activate" ]; then
    echo "$(date): Activating uv/venv environment at $UV_VENV_DIR"
    # shellcheck disable=SC1090
    source "$UV_VENV_DIR/bin/activate"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ] || command -v conda > /dev/null 2>&1; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        eval "$(conda shell.bash hook)"
    fi
    echo "$(date): Activating conda environment ${CONDA_ENV:-lmms-eval}"
    conda activate "${CONDA_ENV:-lmms-eval}"
else
    echo "ERROR: No uv/venv found at $UV_VENV_DIR and conda is not available."
    exit 1
fi

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
# Use torch to count GPUs — nvidia-smi -L ignores CUDA_VISIBLE_DEVICES and
# would return all GPUs on the node, not just the ones allocated to this job.
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
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
echo "  Tasks:       $REQUESTED_TASKS"
echo "  Batch size:  $BATCH_SIZE"
echo "  GPUs:        $NUM_GPUS"
echo "  Results dir: $RESULTS_DIR/$MODEL_NAME"

mkdir -p "$RESULTS_DIR" logs

# Run only tasks that are still missing outputs.
PENDING_TASKS="$(pending_tasks_csv "$RESULTS_DIR/$MODEL_NAME" "$REQUESTED_TASKS")"
if [ -z "$PENDING_TASKS" ]; then
    echo "$(date): Requested task outputs already exist for $MODEL_NAME ($REQUESTED_TASKS), skipping."
    exit 0
fi

if [ "$PENDING_TASKS" != "$REQUESTED_TASKS" ]; then
    echo "$(date): Some task outputs already exist; running only missing tasks: $PENDING_TASKS"
fi

cd "$LMMS_EVAL_DIR"
python -m lmms_eval \
    --model "$MODEL_TYPE" \
    --model_args "$MODEL_ARGS" \
    --tasks "$PENDING_TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$RESULTS_DIR/$MODEL_NAME" \
    --log_samples \
    --force_simple \
    --verbosity INFO

echo "$(date): Evaluation complete. Results in $RESULTS_DIR/$MODEL_NAME"
