#!/bin/bash
#
# Submit lmms-eval SLURM jobs for all model checkpoints.
#
# Usage:
#   ./scripts/submit_all.sh                           # all models, all tasks, mcml partitions
#   ./scripts/submit_all.sh internvl3                 # filter to InternVL3 family
#   ./scripts/submit_all.sh --dry-run                 # print sbatch commands without submitting
#   ./scripts/submit_all.sh --tasks vlind_bench_oe    # single benchmark
#   ./scripts/submit_all.sh --tasks vlind_bench_oe --time 0:30:00 --time-large 1:30:00
#
# To maximise scheduling throughput on a busy cluster, run submit_all.sh three times
# targeting different partition pools. The pending_tasks_csv guard in slurm_eval.sh
# ensures that whichever job starts first does the work; the others exit immediately.
#
#   # Wave 1: mcml A100/H100
#   ./scripts/submit_all.sh --tasks vlind_bench_oe --time 0:30:00 --time-large 1:30:00
#
#   # Wave 2: LRZ A100/H100 (qos=gpu required for lrz partitions)
#   ./scripts/submit_all.sh --tasks vlind_bench_oe --time 0:30:00 --time-large 1:30:00 \
#     --partition lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8 --qos gpu
#
#   # Wave 3: A100 MIG slices (40 GB, 1-GPU models only, qos=mig)
#   ./scripts/submit_all.sh --tasks vlind_bench_oe --time 0:30:00 --mig

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/slurm_eval.sh"

FAMILY_FILTER=""
DRY_RUN=false
MIG_MODE=false
# Resolved once here and forwarded explicitly to each sbatch job.
# Intentionally ignore externally exported TASKS to avoid accidental overrides.
TASKS="vilp_without_fact,vilp,vlms_are_biased,vlind_bench,vlind_bench_oe"
# Time limits: tune these down when running only fast benchmarks (e.g. vlind_bench_oe)
# to improve backfill scheduling on busy clusters.
#   Full 5-task suite:  TIME_SMALL=2:00:00  TIME_LARGE=6:00:00
#   vlind_bench_oe only: TIME_SMALL=0:30:00  TIME_LARGE=1:30:00
TIME_SMALL="2:00:00"
TIME_LARGE="6:00:00"
# Partition and QOS.
#   mcml partitions require --qos=mcml
#   lrz partitions require  --qos=gpu
#   mig partitions require  --qos=mig   (set automatically by --mig)
PARTITION="mcml-hgx-a100-80x4,mcml-hgx-h100-94x4"
QOS="mcml"

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --mig)
            # MIG mode: target the mcml 40 GB A100 MIG partition.
            # Uses gpu:3g.40gb:1 GRES; skips 4-GPU models (too large for a single slice).
            MIG_MODE=true
            PARTITION="mcml-hgx-a100-80x4-mig"
            QOS="mig"
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
        --time)
            if [ $# -lt 2 ]; then
                echo "ERROR: --time requires a value (e.g. 0:30:00)"
                exit 1
            fi
            TIME_SMALL="$2"
            shift 2
            ;;
        --time=*)
            TIME_SMALL="${1#--time=}"
            shift
            ;;
        --time-large)
            if [ $# -lt 2 ]; then
                echo "ERROR: --time-large requires a value (e.g. 1:30:00)"
                exit 1
            fi
            TIME_LARGE="$2"
            shift 2
            ;;
        --time-large=*)
            TIME_LARGE="${1#--time-large=}"
            shift
            ;;
        --partition)
            if [ $# -lt 2 ]; then
                echo "ERROR: --partition requires a value"
                exit 1
            fi
            PARTITION="$2"
            shift 2
            ;;
        --partition=*)
            PARTITION="${1#--partition=}"
            shift
            ;;
        --qos)
            if [ $# -lt 2 ]; then
                echo "ERROR: --qos requires a value"
                exit 1
            fi
            QOS="$2"
            shift 2
            ;;
        --qos=*)
            QOS="${1#--qos=}"
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

# In MIG mode only 1-GPU models fit in a single 40 GB slice; skip the 4-GPU pass.
GPU_PASSES=(1 4)
if [ "$MIG_MODE" = true ]; then
    GPU_PASSES=(1)
    echo "MIG mode: submitting 1-GPU models only (gres=gpu:3g.40gb:1, partition=$PARTITION, qos=$QOS)"
fi

# Submit 1-GPU jobs first (faster backfill), then multi-GPU
for pass in "${GPU_PASSES[@]}"; do
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
            if [ "$MIG_MODE" = true ]; then
                # MIG slice: 40 GB A100 partition, specific GRES type required
                RESOURCES=(
                    --gres=gpu:3g.40gb:1
                    --cpus-per-task=8
                    --mem=80G
                    --time="$TIME_SMALL"
                    --partition="$PARTITION"
                    --qos="$QOS"
                )
            else
                # Small models: tight resource requests for fast backfill
                RESOURCES=(
                    --gres=gpu:1
                    --cpus-per-task=8
                    --mem=80G
                    --time="$TIME_SMALL"
                    --partition="$PARTITION"
                    --qos="$QOS"
                )
            fi
        else
            # Large models: 4 GPUs
            RESOURCES=(
                --gres=gpu:4
                --cpus-per-task=32
                --mem=320G
                --time="$TIME_LARGE"
                --partition="$PARTITION"
                --qos="$QOS"
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
