#!/bin/bash
#
# Submit lmms-eval SLURM jobs on JUWELS.
#
# JUWELS allocates full nodes: 4× H200 (100 GB), 288 cores each.
# All jobs must use all 4 GPUs:
#   - 1-GPU models: batched 4-per-node, run in parallel via srun
#   - 4-GPU models: 1 model per node, device_map=auto
#
# Usage:
#   ./scripts/submit_all.sh                           # all models, all tasks
#   ./scripts/submit_all.sh internvl3                 # filter to InternVL3 family
#   ./scripts/submit_all.sh --dry-run
#   ./scripts/submit_all.sh --tasks vlind_bench_oe
#   ./scripts/submit_all.sh --tasks vlind_bench_oe --time 0:30:00 --time-large 1:30:00

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/slurm_eval.sh"

FAMILY_FILTER=""
DRY_RUN=false
TASKS="vilp_without_fact,vilp,vlms_are_biased,vlind_bench,vlind_bench_oe"
# Time limits: lower these when running only fast benchmarks to improve backfill scheduling.
#   Full suite:        TIME_SMALL=2:00:00  TIME_LARGE=6:00:00
#   vlind_bench_oe:    TIME_SMALL=0:30:00  TIME_LARGE=1:30:00
TIME_SMALL="2:00:00"
TIME_LARGE="6:00:00"
ACCOUNT="taco-vlm"
PARTITION="gpus"
RESULTS_DIR="$SCRATCH/grob1/results/lmms-eval"

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --tasks=*) TASKS="${1#--tasks=}"; shift ;;
        --time) TIME_SMALL="$2"; shift 2 ;;
        --time=*) TIME_SMALL="${1#--time=}"; shift ;;
        --time-large) TIME_LARGE="$2"; shift 2 ;;
        --time-large=*) TIME_LARGE="${1#--time-large=}"; shift ;;
        *) FAMILY_FILTER="$1"; shift ;;
    esac
done

# ---- Model definitions ----
# Format: family|model_type|pretrained_id|model_name|num_gpus|batch_size
# num_gpus=1: batched 4-per-node; num_gpus=4: 1-per-node with device_map=auto
MODELS=(
    # InternVL3.5
    "internvl3_5|internvl3_5|OpenGVLab/InternVL3_5-1B|internvl3_5-1b|1|1"
    "internvl3_5|internvl3_5|OpenGVLab/InternVL3_5-2B|internvl3_5-2b|1|1"
    "internvl3_5|internvl3_5|OpenGVLab/InternVL3_5-4B|internvl3_5-4b|1|1"
    "internvl3_5|internvl3_5|OpenGVLab/InternVL3_5-8B|internvl3_5-8b|1|1"
    "internvl3_5|internvl3_5|OpenGVLab/InternVL3_5-14B|internvl3_5-14b|1|1"
    "internvl3_5|internvl3_5|OpenGVLab/InternVL3_5-38B|internvl3_5-38b|4|1"

    # InternVL3
    "internvl3|internvl3|OpenGVLab/InternVL3-1B|internvl3-1b|1|1"
    "internvl3|internvl3|OpenGVLab/InternVL3-2B|internvl3-2b|1|1"
    "internvl3|internvl3|OpenGVLab/InternVL3-9B|internvl3-9b|1|1"
    "internvl3|internvl3|OpenGVLab/InternVL3-8B|internvl3-8b|1|1"
    "internvl3|internvl3|OpenGVLab/InternVL3-14B|internvl3-14b|1|1"
    "internvl3|internvl3|OpenGVLab/InternVL3-38B|internvl3-38b|4|1"
    "internvl3|internvl3|OpenGVLab/InternVL3-78B|internvl3-78b|4|1"

    # Qwen2.5-VL
    "qwen25vl|qwen2_5_vl|Qwen/Qwen2.5-VL-3B-Instruct|qwen25vl-3b|1|1"
    "qwen25vl|qwen2_5_vl|Qwen/Qwen2.5-VL-7B-Instruct|qwen25vl-7b|1|1"
    "qwen25vl|qwen2_5_vl|Qwen/Qwen2.5-VL-32B-Instruct|qwen25vl-32b|4|1"
    "qwen25vl|qwen2_5_vl|Qwen/Qwen2.5-VL-72B-Instruct|qwen25vl-72b|4|1"

    # Qwen3-VL
    "qwen3vl|qwen3_vl|Qwen/Qwen3-VL-2B-Instruct|qwen3vl-2b|1|1"
    "qwen3vl|qwen3_vl|Qwen/Qwen3-VL-4B-Instruct|qwen3vl-4b|1|1"
    "qwen3vl|qwen3_vl|Qwen/Qwen3-VL-8B-Instruct|qwen3vl-8b|1|1"
    "qwen3vl|qwen3_vl|Qwen/Qwen3-VL-32B-Instruct|qwen3vl-32b|4|1"

    # LLaVA-OneVision
    "llava_ov|llava_onevision|lmms-lab/llava-onevision-qwen2-0.5b-ov|llava-ov-0.5b|1|1"
    "llava_ov|llava_onevision|lmms-lab/llava-onevision-qwen2-7b-ov|llava-ov-7b|1|1"
    "llava_ov|llava_onevision|lmms-lab/llava-onevision-qwen2-72b-ov|llava-ov-72b|4|1"

    # Gemma-3
    "gemma3|gemma3|google/gemma-3-4b-it|gemma3-4b|1|1"
    "gemma3|gemma3|google/gemma-3-12b-it|gemma3-12b|1|1"
    "gemma3|gemma3|google/gemma-3-27b-it|gemma3-27b|4|1"

    # LLaVA 1.5
    "llava15|llava|liuhaotian/llava-v1.5-7b|llava15-7b|1|1"
    "llava15|llava|liuhaotian/llava-v1.5-13b|llava15-13b|1|1"
)

task_has_output() {
    local model_results_dir="$1" task="$2"
    compgen -G "$model_results_dir"/*/*_samples_"$task".jsonl > /dev/null 2>&1
}

all_tasks_complete() {
    local model_results_dir="$1" tasks_csv="$2"
    IFS=',' read -r -a task_list <<< "$tasks_csv"
    for task in "${task_list[@]}"; do
        task="${task#"${task%%[![:space:]]*}"}"
        task="${task%"${task##*[![:space:]]}"}"
        task_has_output "$model_results_dir" "$task" || return 1
    done
    return 0
}

# Submits a batch of 1–4 model entries as one node job.
# Each entry occupies one GPU slot via srun in slurm_eval.sh.
# Uses bash nameref (requires bash 4.3+).
submit_batch() {
    local -n _entries="$1"
    local time="$2" num_gpus="$3"

    [ ${#_entries[@]} -eq 0 ] && return

    IFS='|' read -r _ _ _ first_name _ _ <<< "${_entries[0]}"
    local job_name="eval-${first_name}"
    [ ${#_entries[@]} -gt 1 ] && job_name="${job_name}+$((${#_entries[@]}-1))"

    # sbatch --export is comma-delimited; pass each model's vars separately.
    local export_vars="ALL,TASKS=${TASKS_FOR_EXPORT},NUM_GPUS=${num_gpus}"
    for i in "${!_entries[@]}"; do
        local n=$((i+1))
        IFS='|' read -r _ model_type pretrained model_name _ batch_size <<< "${_entries[$i]}"
        export_vars="${export_vars},MODEL_TYPE_${n}=${model_type}"
        export_vars="${export_vars},PRETRAINED_${n}=${pretrained}"
        export_vars="${export_vars},MODEL_NAME_${n}=${model_name}"
        export_vars="${export_vars},BATCH_SIZE_${n}=${batch_size}"
    done

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] sbatch --account=$ACCOUNT --nodes=1 --partition=$PARTITION --gres=gpu:4 --time=$time --job-name=$job_name $SLURM_SCRIPT"
        echo "          export: $export_vars"
    else
        echo "Submitting: $job_name (${#_entries[@]} model(s), ${num_gpus}-GPU each)"
        sbatch \
            --account="$ACCOUNT" \
            --nodes=1 \
            --partition="$PARTITION" \
            --gres=gpu:4 \
            --time="$time" \
            --job-name="$job_name" \
            --export="$export_vars" \
            "$SLURM_SCRIPT"
    fi
}

TASKS_FOR_EXPORT="${TASKS//,/;}"
SUBMITTED_NODES=0
SKIPPED=0

# ---- 1-GPU models: batch 4 per node ----
SMALL_BATCH=()
for entry in "${MODELS[@]}"; do
    IFS='|' read -r family _ _ model_name num_gpus _ <<< "$entry"
    [ -n "$FAMILY_FILTER" ] && [ "$family" != "$FAMILY_FILTER" ] && continue
    [ "$num_gpus" -ne 1 ] && continue

    if all_tasks_complete "$RESULTS_DIR/$model_name" "$TASKS"; then
        echo "Skipping: $model_name (all tasks complete)"
        SKIPPED=$((SKIPPED+1))
        continue
    fi

    SMALL_BATCH+=("$entry")
    if [ ${#SMALL_BATCH[@]} -eq 4 ]; then
        submit_batch SMALL_BATCH "$TIME_SMALL" 1
        SUBMITTED_NODES=$((SUBMITTED_NODES+1))
        SMALL_BATCH=()
    fi
done
# Flush remainder — still requires a full node even if fewer than 4 models.
if [ ${#SMALL_BATCH[@]} -gt 0 ]; then
    submit_batch SMALL_BATCH "$TIME_SMALL" 1
    SUBMITTED_NODES=$((SUBMITTED_NODES+1))
fi

# ---- 4-GPU models: one per node ----
for entry in "${MODELS[@]}"; do
    IFS='|' read -r family _ _ model_name num_gpus _ <<< "$entry"
    [ -n "$FAMILY_FILTER" ] && [ "$family" != "$FAMILY_FILTER" ] && continue
    [ "$num_gpus" -ne 4 ] && continue

    if all_tasks_complete "$RESULTS_DIR/$model_name" "$TASKS"; then
        echo "Skipping: $model_name (all tasks complete)"
        SKIPPED=$((SKIPPED+1))
        continue
    fi

    LARGE_BATCH=("$entry")
    submit_batch LARGE_BATCH "$TIME_LARGE" 4
    SUBMITTED_NODES=$((SUBMITTED_NODES+1))
done

echo ""
echo "Submitted $SUBMITTED_NODES node-job(s), skipped $SKIPPED model(s) (already complete)."
