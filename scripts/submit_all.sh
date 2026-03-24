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

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) FAMILY_FILTER="$arg" ;;
    esac
done

# ---- Model definitions ----
# Format: family|model_type|pretrained_id|model_name|num_gpus
MODELS=(
    # InternVL3
    "internvl3|internvl3|OpenGVLab/InternVL3-1B|internvl3-1b|1"
    "internvl3|internvl3|OpenGVLab/InternVL3-2B|internvl3-2b|1"
    "internvl3|internvl3|OpenGVLab/InternVL3-4B|internvl3-4b|1"
    "internvl3|internvl3|OpenGVLab/InternVL3-8B|internvl3-8b|1"
    "internvl3|internvl3|OpenGVLab/InternVL3-14B|internvl3-14b|1"
    "internvl3|internvl3|OpenGVLab/InternVL3-38B|internvl3-38b|4"
    "internvl3|internvl3|OpenGVLab/InternVL3-78B|internvl3-78b|4"

    # Qwen2.5-VL
    "qwen25vl|qwen2_5_vl|Qwen/Qwen2.5-VL-3B-Instruct|qwen25vl-3b|1"
    "qwen25vl|qwen2_5_vl|Qwen/Qwen2.5-VL-7B-Instruct|qwen25vl-7b|1"
    "qwen25vl|qwen2_5_vl|Qwen/Qwen2.5-VL-32B-Instruct|qwen25vl-32b|4"
    "qwen25vl|qwen2_5_vl|Qwen/Qwen2.5-VL-72B-Instruct|qwen25vl-72b|4"

    # Qwen3-VL
    "qwen3vl|qwen3_vl|Qwen/Qwen3-VL-2B|qwen3vl-2b|1"
    "qwen3vl|qwen3_vl|Qwen/Qwen3-VL-8B|qwen3vl-8b|1"

    # LLaVA-OneVision
    "llava_ov|llava_onevision|lmms-lab/llava-onevision-qwen2-0.5b-ov|llava-ov-0.5b|1"
    "llava_ov|llava_onevision|lmms-lab/llava-onevision-qwen2-7b-ov|llava-ov-7b|1"
    "llava_ov|llava_onevision|lmms-lab/llava-onevision-qwen2-72b-ov|llava-ov-72b|4"

    # Gemma-3
    "gemma3|gemma3|google/gemma-3-4b-it|gemma3-4b|1"
    "gemma3|gemma3|google/gemma-3-12b-it|gemma3-12b|1"
    "gemma3|gemma3|google/gemma-3-27b-it|gemma3-27b|4"

    # LLaVA 1.5
    "llava15|llava|liuhaotian/llava-v1.5-7b|llava15-7b|1"
    "llava15|llava|liuhaotian/llava-v1.5-13b|llava15-13b|1"
)

# ---- Resource tiers ----
get_resources() {
    local num_gpus=$1
    local model_name=$2

    if [ "$num_gpus" -eq 1 ]; then
        # Small models: tight resource requests for fast backfill
        echo "--gres=gpu:h100:1 --cpus-per-task=8 --mem=80G --time=03:00:00"
    else
        # Large models: 4x H100
        echo "--gres=gpu:h100:4 --cpus-per-task=32 --mem=320G --time=08:00:00"
    fi
}

# ---- Submit jobs ----
SUBMITTED=0
SKIPPED=0

# Submit 1-GPU jobs first (faster backfill), then multi-GPU
for pass in 1 4; do
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r family model_type pretrained model_name num_gpus <<< "$entry"

        # Filter by family if specified
        if [ -n "$FAMILY_FILTER" ] && [ "$family" != "$FAMILY_FILTER" ]; then
            continue
        fi

        # This pass only handles jobs with matching GPU count
        if [ "$num_gpus" -ne "$pass" ]; then
            continue
        fi

        RESOURCES=$(get_resources "$num_gpus" "$model_name")
        JOB_NAME="eval-${model_name}"

        CMD="sbatch --job-name=$JOB_NAME $RESOURCES \
            --export=ALL,MODEL_TYPE=$model_type,PRETRAINED=$pretrained,MODEL_NAME=$model_name \
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
echo "Submitted $SUBMITTED jobs."
