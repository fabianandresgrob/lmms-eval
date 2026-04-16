#!/bin/bash
#
# Submit lmms-eval SLURM jobs for all model checkpoints.
#
# Usage:
#   ./scripts/submit_all.sh              # submit all models
#   ./scripts/submit_all.sh internvl3    # submit only InternVL3 family
#   ./scripts/submit_all.sh --dry-run    # print commands without submitting
#   ./scripts/submit_all.sh --tasks vilp_without_fact,vilp,vlms_are_biased,vlind_bench
#
# Each job evaluates one model checkpoint on all benchmarks (vlms_are_biased, vilp, vlind_bench).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/slurm_eval.sh"

FAMILY_FILTER=""
DRY_RUN=false
# Resolved once here and forwarded explicitly to each sbatch job.
# Intentionally ignore externally exported TASKS to avoid accidental overrides.
TASKS="vilp_without_fact,vilp,vlms_are_biased,vlind_bench"

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --tasks)
            if [ $# -lt 2 ]; then
                echo "ERROR: --tasks requires a comma-separated value"
                exit 1
            fi
            TASKS="$2"
            shift 2
            ;;
        --tasks=*)
            TASKS="${1#--tasks=}"
            shift
            ;;
        *)
            FAMILY_FILTER="$1"
            shift
            ;;
    esac
done

# ---- Model definitions ----
# Format: family|model_type|pretrained_id|model_name|num_gpus|batch_size|conda_env
# batch_size=1 for all: cross-task batching bug causes IndexError when batch_size>1
# with multiple tasks running in the same job.
# conda_env is optional (defaults to "lmms-eval").
MODELS=(
    # InternVL3.5
    "internvl3_5|internvl3_5|OpenGVLab/InternVL3_5-1B|internvl3_5-1b|1|1"
    "internvl3_5|internvl3_5|OpenGVLab/InternVL3_5-2B|internvl3_5-2b|1|1"
    "internvl3_5|internvl3_5|OpenGVLab/InternVL3_5-4B|internvl3_5-4b|1|1"
    "internvl3_5|internvl3_5|OpenGVLab/InternVL3_5-8B|internvl3_5-8b|1|1"
    "internvl3_5|internvl3_5|OpenGVLab/InternVL3_5-14B|internvl3_5-14b|1|1"
    "internvl3_5|internvl3_5|OpenGVLab/InternVL3_5-38B|internvl3_5-38b|4|1"

    # InternVL3 (batch_size=1 enforced by model)
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

    # LLaVA-OneVision (needs lmms-eval-ov env with LLaVA-NeXT)
    "llava_ov|llava_onevision|lmms-lab/llava-onevision-qwen2-0.5b-ov|llava-ov-0.5b|1|1|lmms-eval-ov"
    "llava_ov|llava_onevision|lmms-lab/llava-onevision-qwen2-7b-ov|llava-ov-7b|1|1|lmms-eval-ov"
    "llava_ov|llava_onevision|lmms-lab/llava-onevision-qwen2-72b-ov|llava-ov-72b|4|1|lmms-eval-ov"

    # Gemma-3 (batch_size>1 breaks when running multiple tasks)
    "gemma3|gemma3|google/gemma-3-4b-it|gemma3-4b|1|1"
    "gemma3|gemma3|google/gemma-3-12b-it|gemma3-12b|1|1"
    "gemma3|gemma3|google/gemma-3-27b-it|gemma3-27b|4|1"

    # LLaVA 1.5 (same cross-task batching bug)
    "llava15|llava|liuhaotian/llava-v1.5-7b|llava15-7b|1|1"
    "llava15|llava|liuhaotian/llava-v1.5-13b|llava15-13b|1|1"
)

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

# ---- Submit jobs ----
SUBMITTED=0
SKIPPED=0

# Submit 1-GPU jobs first (faster backfill), then multi-GPU
for pass in 1 4; do
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r family model_type pretrained model_name num_gpus batch_size conda_env <<< "$entry"

        # Filter by family if specified
        if [ -n "$FAMILY_FILTER" ] && [ "$family" != "$FAMILY_FILTER" ]; then
            continue
        fi

        # This pass only handles jobs with matching GPU count
        if [ "$num_gpus" -ne "$pass" ]; then
            continue
        fi

        # Skip only if all requested task outputs already exist
        RESULTS_DIR="$SCRATCH/results/lmms-eval"
        if all_tasks_complete "$RESULTS_DIR/$model_name" "$TASKS"; then
            echo "Skipping: $model_name (all requested task outputs already exist: $TASKS)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        if [ "$num_gpus" -eq 1 ]; then
            # Small models: tight resource requests for fast backfill
            RESOURCES=(
                --gres=gpu:1
                --cpus-per-task=8
                --mem=80G
                --time=03:00:00
            )
        else
            # Large models: 4 GPUs
            RESOURCES=(
                --gres=gpu:4
                --cpus-per-task=32
                --mem=320G
                --time=08:00:00
            )
        fi

        JOB_NAME="eval-${model_name}"

        # NOTE: sbatch --export is comma-delimited. If TASKS contains commas,
        # it gets truncated at the first task. Encode commas as ';' here and
        # decode back in slurm_eval.sh.
        TASKS_FOR_EXPORT="${TASKS//,/;}"
        EXPORT_VARS="ALL,MODEL_TYPE=$model_type,PRETRAINED=$pretrained,MODEL_NAME=$model_name,BATCH_SIZE=$batch_size,TASKS=$TASKS_FOR_EXPORT"
        if [ -n "$conda_env" ]; then
            EXPORT_VARS="$EXPORT_VARS,CONDA_ENV=$conda_env"
        fi

        if [ "$DRY_RUN" = true ]; then
            printf "[DRY RUN] sbatch --job-name=%q " "$JOB_NAME"
            printf "%q " "${RESOURCES[@]}"
            printf -- "--export=%q %q\n" "$EXPORT_VARS" "$SLURM_SCRIPT"
        else
            echo "Submitting: $JOB_NAME ($num_gpus GPU)"
            sbatch --job-name="$JOB_NAME" "${RESOURCES[@]}" --export="$EXPORT_VARS" "$SLURM_SCRIPT"
        fi
        SUBMITTED=$((SUBMITTED + 1))
    done
done

echo ""
echo "Submitted $SUBMITTED jobs, skipped $SKIPPED (already complete)."
