#!/bin/bash
#SBATCH --account=taco-vlm
#SBATCH --nodes=1
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --time=6:00:00
#SBATCH --job-name=lmms-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

# Adjust to where the repo is cloned on JUWELS.
LMMS_EVAL_DIR="$PROJECT/grob1/lmms-eval"
RESULTS_DIR="$SCRATCH/grob1/results/lmms-eval"
# submit_all.sh encodes commas as ';' because sbatch --export is comma-delimited.
REQUESTED_TASKS_RAW="${TASKS:-vlms_are_biased,vilp,vlind_bench,vlind_bench_oe}"
REQUESTED_TASKS="${REQUESTED_TASKS_RAW//;/,}"

# ---- Helper functions ----

task_has_output() {
    local model_results_dir="$1" task="$2"
    compgen -G "$model_results_dir"/*/*_samples_"$task".jsonl > /dev/null 2>&1
}

pending_tasks_csv() {
    local model_results_dir="$1" tasks_csv="$2"
    local pending=""
    IFS=',' read -r -a task_list <<< "$tasks_csv"
    for task in "${task_list[@]}"; do
        task="${task#"${task%%[![:space:]]*}"}"
        task="${task%"${task##*[![:space:]]}"}"
        [ -z "$task" ] && continue
        if ! task_has_output "$model_results_dir" "$task"; then
            pending="${pending:+$pending,}$task"
        fi
    done
    echo "$pending"
}

run_model() {
    local model_type="$1" pretrained="$2" model_name="$3" batch_size="${4:-1}" num_gpus="${5:-1}"
    local model_results_dir="$RESULTS_DIR/$model_name"

    mkdir -p "$model_results_dir" logs

    # mkdir is atomic on shared filesystems — prevents two concurrent workers
    # from running the same model.
    local lock_dir="$model_results_dir/.lock"
    if ! mkdir "$lock_dir" 2>/dev/null; then
        echo "$(date): Lock held for $model_name — skipping."
        return 0
    fi
    trap 'rmdir "'"$lock_dir"'" 2>/dev/null || true' RETURN INT TERM EXIT

    local pending
    pending="$(pending_tasks_csv "$model_results_dir" "$REQUESTED_TASKS")"
    if [ -z "$pending" ]; then
        echo "$(date): $model_name — all tasks complete, skipping."
        return 0
    fi
    [ "$pending" != "$REQUESTED_TASKS" ] && \
        echo "$(date): $model_name — running only missing tasks: $pending"

    local model_args="pretrained=$pretrained"
    [ "$num_gpus" -gt 1 ] && model_args="${model_args},device_map=auto"

    echo "$(date): Starting $model_name (${num_gpus} GPU(s)), tasks: $pending"
    cd "$LMMS_EVAL_DIR"
    python -m lmms_eval \
        --model "$model_type" \
        --model_args "$model_args" \
        --tasks "$pending" \
        --batch_size "$batch_size" \
        --output_path "$model_results_dir" \
        --log_samples \
        --force_simple \
        --verbosity INFO
    echo "$(date): $model_name complete."
}

# ---- Worker mode ----
# Called via: srun bash slurm_eval.sh --worker N
# Reads MODEL_TYPE_N, PRETRAINED_N, MODEL_NAME_N, BATCH_SIZE_N from the
# inherited SLURM environment. CUDA_VISIBLE_DEVICES is set by srun per GPU slot.
if [ "$1" = "--worker" ]; then
    N="$2"
    model_type_var="MODEL_TYPE_$N"
    pretrained_var="PRETRAINED_$N"
    model_name_var="MODEL_NAME_$N"
    batch_size_var="BATCH_SIZE_$N"
    run_model "${!model_type_var}" "${!pretrained_var}" "${!model_name_var}" "${!batch_size_var:-1}" 1
    exit 0
fi

# ---- Main job ----

if [ -z "$MODEL_TYPE_1" ] || [ -z "$PRETRAINED_1" ] || [ -z "$MODEL_NAME_1" ]; then
    echo "ERROR: MODEL_TYPE_1, PRETRAINED_1, MODEL_NAME_1 must be set."
    exit 1
fi

# Keep datasets cache on scratch — lmms-eval may redirect to /tmp otherwise.
export LMMS_EVAL_DATASETS_CACHE="$SCRATCH/grob1/.cache/huggingface/datasets"

# Compute nodes have no internet — force HuggingFace/transformers to use cache only.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export PATH="${CUDA_HOME}/bin:${PATH}"

echo "$(date): Activating environment"
source "${ACTIVATE_SCRIPT:-$LMMS_EVAL_DIR/sc_venv_template/activate.sh}"
[ -n "${PYTHONPATH_EXTRA:-}" ] && export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"

nvidia-smi

NUM_GPUS="${NUM_GPUS:-1}"
if [ "$NUM_GPUS" -gt 1 ]; then
    # Single large model using all 4 GPUs with device_map=auto.
    run_model "$MODEL_TYPE_1" "$PRETRAINED_1" "$MODEL_NAME_1" "${BATCH_SIZE_1:-1}" "$NUM_GPUS"
else
    # Up to 4 small models in parallel, each pinned to one GPU via srun.
    SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
    for i in 1 2 3 4; do
        pretrained_var="PRETRAINED_$i"
        [ -z "${!pretrained_var}" ] && continue
        srun --exclusive -n 1 --gres=gpu:1 \
            --output="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu${i}.out" \
            --error="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu${i}.err" \
            bash "$SCRIPT_PATH" --worker "$i" &
    done
    wait
fi
