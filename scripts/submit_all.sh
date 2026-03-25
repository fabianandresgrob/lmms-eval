#!/bin/bash
#
# Submit lmms-eval SLURM jobs for all model checkpoints.
#
# Usage:
#   ./scripts/submit_all.sh              # submit all models
#   ./scripts/submit_all.sh internvl3    # submit only InternVL3 family
#   ./scripts/submit_all.sh --dry-run    # print commands without submitting
#
# Each job evaluates one model checkpoint on all benchmarks (vlms_are_biased, vilp, vlind_bench).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/slurm_eval.sh"

FAMILY_FILTER=""
DRY_RUN=false

# Only schedule on nodes with >=80GB VRAM (A100 80GB or H100 80GB).
# This avoids V100 32GB, A100 40GB, A100 MIG 20GB, and RTX 8000 48GB nodes.
GPU_CONSTRAINT="a100_80gb|h100_80gb"

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) FAMILY_FILTER="$arg" ;;
    esac
done

# ---- Model definitions ----
# Format: family|model_type|pretrained_id|model_name|num_gpus|batch_size|conda_env
# batch_size=1 for all: cross-task batching bug causes IndexError when batch_size>1
# with multiple tasks running in the same job.
# conda_env is optional (defaults to "lmms-eval").
MODELS=(
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

# ---- Resource tiers ----
get_resources() {
    local num_gpus=$1
    local model_name=$2

    if [ "$num_gpus" -eq 1 ]; then
        # Small models: tight resource requests for fast backfill
        echo "--gres=gpu:1 --cpus-per-task=8 --mem=80G --time=03:00:00 --constraint=\"$GPU_CONSTRAINT\""
    else
        # Large models: 4 GPUs
        echo "--gres=gpu:4 --cpus-per-task=32 --mem=320G --time=08:00:00 --constraint=\"$GPU_CONSTRAINT\""
    fi
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

        # Skip if results already exist
        RESULTS_DIR="$SCRATCH/results/lmms-eval"
        if compgen -G "$RESULTS_DIR/$model_name"/*/*_results.json > /dev/null 2>&1; then
            echo "Skipping: $model_name (results already exist)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        RESOURCES=$(get_resources "$num_gpus" "$model_name")
        JOB_NAME="eval-${model_name}"

        EXPORT_VARS="ALL,MODEL_TYPE=$model_type,PRETRAINED=$pretrained,MODEL_NAME=$model_name,BATCH_SIZE=$batch_size"
        if [ -n "$conda_env" ]; then
            EXPORT_VARS="$EXPORT_VARS,CONDA_ENV=$conda_env"
        fi

        CMD="sbatch --job-name=$JOB_NAME $RESOURCES \
            --export=$EXPORT_VARS \
            $SLURM_SCRIPT"

        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] $CMD"
        else
            echo "Submitting: $JOB_NAME ($num_gpus GPU)"
            eval "$CMD"
        fi
        SUBMITTED=$((SUBMITTED + 1))
    done
done

echo ""
echo "Submitted $SUBMITTED jobs, skipped $SKIPPED (already complete)."
