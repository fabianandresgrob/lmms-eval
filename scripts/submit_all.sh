#!/bin/bash
#
# Submit lmms-eval SLURM jobs on JUWELS.
#
# JUWELS allocates full nodes: 4× H200 (100 GB), 288 cores each.
# All jobs must use all 4 GPUs:
#   - 1-GPU models: batched 4-per-node, run in parallel via srun
#   - 4-GPU models: 1 model per node, device_map=auto
#
# Multiple Python envs are supported via the 7th field in MODELS.
# Only models sharing the same env are batched together on one node.
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

# Root of all project repos on JUWELS — adjust if your layout differs.
REPOS_DIR="${PROJECT}/grob1"

FAMILY_FILTER=""
DRY_RUN=false
TASKS="vilp_without_fact,vilp,vlms_are_biased,vlind_bench,vlind_bench_oe"
# Time limits: lower these when running only fast benchmarks to improve backfill scheduling.
#   Full suite:        TIME_SMALL=2:00:00  TIME_LARGE=6:00:00
#   vlind_bench_oe:    TIME_SMALL=0:30:00  TIME_LARGE=1:30:00
TIME_SMALL="2:00:00"
TIME_LARGE="6:00:00"
ACCOUNT="taco-vlm"
PARTITION="booster"
RESULTS_DIR="$SCRATCH/grob1/results/lmms-eval"

# Env activate.sh paths — keep these in one place.
ENV_MAIN="${REPOS_DIR}/lmms-eval/sc_venv_template/activate.sh"
ENV_LLAVA="${REPOS_DIR}/lmms-eval/sc_venv_template-llava/activate.sh"   # LLaVA-NeXT — LLaVA OV only
ENV_SAE_LLAVA="${REPOS_DIR}/LLaVA/sc_venv_template/activate.sh"           # SAE-modified LLaVA 1.5 — SAE models + LLaVA 1.5 baseline

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
# Format: family|model_type|pretrained_id|model_name|num_gpus|batch_size|activate_script
# activate_script: path to sc_venv_template/activate.sh for this model's env.
#   Leave empty to use the default main env (ENV_MAIN).
#   Only models with the same activate_script are batched onto the same node.
MODELS=(
    # InternVL3.5
    "internvl3_5|internvl3_5|OpenGVLab/InternVL3_5-1B|internvl3_5-1b|1|1|"
    "internvl3_5|internvl3_5|OpenGVLab/InternVL3_5-2B|internvl3_5-2b|1|1|"
    "internvl3_5|internvl3_5|OpenGVLab/InternVL3_5-4B|internvl3_5-4b|1|1|"
    "internvl3_5|internvl3_5|OpenGVLab/InternVL3_5-8B|internvl3_5-8b|1|1|"
    "internvl3_5|internvl3_5|OpenGVLab/InternVL3_5-14B|internvl3_5-14b|1|1|"
    "internvl3_5|internvl3_5|OpenGVLab/InternVL3_5-38B|internvl3_5-38b|1|1|"

    # InternVL3
    "internvl3|internvl3|OpenGVLab/InternVL3-1B|internvl3-1b|1|1|"
    "internvl3|internvl3|OpenGVLab/InternVL3-2B|internvl3-2b|1|1|"
    "internvl3|internvl3|OpenGVLab/InternVL3-9B|internvl3-9b|1|1|"
    "internvl3|internvl3|OpenGVLab/InternVL3-8B|internvl3-8b|1|1|"
    "internvl3|internvl3|OpenGVLab/InternVL3-14B|internvl3-14b|1|1|"
    "internvl3|internvl3|OpenGVLab/InternVL3-38B|internvl3-38b|1|1|"
    "internvl3|internvl3|OpenGVLab/InternVL3-78B|internvl3-78b|4|1|"

    # Qwen2.5-VL
    "qwen25vl|qwen2_5_vl|Qwen/Qwen2.5-VL-3B-Instruct|qwen25vl-3b|1|1|"
    "qwen25vl|qwen2_5_vl|Qwen/Qwen2.5-VL-7B-Instruct|qwen25vl-7b|1|1|"
    "qwen25vl|qwen2_5_vl|Qwen/Qwen2.5-VL-32B-Instruct|qwen25vl-32b|4|1|"
    "qwen25vl|qwen2_5_vl|Qwen/Qwen2.5-VL-72B-Instruct|qwen25vl-72b|4|1|"

    # Qwen3-VL
    "qwen3vl|qwen3_vl|Qwen/Qwen3-VL-2B-Instruct|qwen3vl-2b|1|1|"
    "qwen3vl|qwen3_vl|Qwen/Qwen3-VL-4B-Instruct|qwen3vl-4b|1|1|"
    "qwen3vl|qwen3_vl|Qwen/Qwen3-VL-8B-Instruct|qwen3vl-8b|1|1|"
    "qwen3vl|qwen3_vl|Qwen/Qwen3-VL-32B-Instruct|qwen3vl-32b|4|1|"

    # LLaVA-OneVision — needs ENV_LLAVA (LLaVA-NeXT + older transformers)
    "llava_ov|llava_onevision|lmms-lab/llava-onevision-qwen2-0.5b-ov|llava-ov-0.5b|1|1|${ENV_LLAVA}"
    "llava_ov|llava_onevision|lmms-lab/llava-onevision-qwen2-7b-ov|llava-ov-7b|1|1|${ENV_LLAVA}"
    "llava_ov|llava_onevision|lmms-lab/llava-onevision-qwen2-72b-ov|llava-ov-72b|4|1|${ENV_LLAVA}"

    # Gemma-3
    "gemma3|gemma3|google/gemma-3-4b-it|gemma3-4b|1|1|"
    "gemma3|gemma3|google/gemma-3-12b-it|gemma3-12b|1|1|"
    "gemma3|gemma3|google/gemma-3-27b-it|gemma3-27b|1|1|"

    # LLaVA 1.5 — uses ENV_SAE_LLAVA (SAE-modified LLaVA fork, compatible with standard LLaVA 1.5)
    "llava15|llava|liuhaotian/llava-v1.5-7b|llava15-7b|1|1|${ENV_SAE_LLAVA}"
    "llava15|llava|liuhaotian/llava-v1.5-13b|llava15-13b|1|1|${ENV_SAE_LLAVA}"

    # LLaVA-MORE Qwen3 — CLIP + Qwen3 with optional SAE bottleneck
    # 0.6B
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-0.6B-baseline|qwen3-0.6B-baseline|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-0.6B-sae-imagenet-enconly|qwen3-0.6B-sae-imagenet-enconly|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-0.6B-sae-imagenet-encdec|qwen3-0.6B-sae-imagenet-encdec|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-0.6B-sae-cc3m_laion-enconly|qwen3-0.6B-sae-cc3m_laion-enconly|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-0.6B-sae-cc3m_laion-encdec|qwen3-0.6B-sae-cc3m_laion-encdec|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-0.6B-sae-llava_ov-enconly|qwen3-0.6B-sae-llava_ov-enconly|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-0.6B-sae-llava_ov-encdec|qwen3-0.6B-sae-llava_ov-encdec|1|1|${ENV_SAE_LLAVA}"
    # 1.7B
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-1.7B-baseline|qwen3-1.7B-baseline|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-1.7B-sae-imagenet-enconly|qwen3-1.7B-sae-imagenet-enconly|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-1.7B-sae-imagenet-encdec|qwen3-1.7B-sae-imagenet-encdec|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-1.7B-sae-cc3m_laion-enconly|qwen3-1.7B-sae-cc3m_laion-enconly|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-1.7B-sae-cc3m_laion-encdec|qwen3-1.7B-sae-cc3m_laion-encdec|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-1.7B-sae-llava_ov-enconly|qwen3-1.7B-sae-llava_ov-enconly|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-1.7B-sae-llava_ov-encdec|qwen3-1.7B-sae-llava_ov-encdec|1|1|${ENV_SAE_LLAVA}"
    # 4B
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-4B-baseline|qwen3-4B-baseline|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-4B-sae-imagenet-enconly|qwen3-4B-sae-imagenet-enconly|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-4B-sae-imagenet-encdec|qwen3-4B-sae-imagenet-encdec|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-4B-sae-cc3m_laion-enconly|qwen3-4B-sae-cc3m_laion-enconly|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-4B-sae-cc3m_laion-encdec|qwen3-4B-sae-cc3m_laion-encdec|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-4B-sae-llava_ov-enconly|qwen3-4B-sae-llava_ov-enconly|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-4B-sae-llava_ov-encdec|qwen3-4B-sae-llava_ov-encdec|1|1|${ENV_SAE_LLAVA}"
    # 8B
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-8B-baseline|qwen3-8B-baseline|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-8B-sae-imagenet-enconly|qwen3-8B-sae-imagenet-enconly|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-8B-sae-imagenet-encdec|qwen3-8B-sae-imagenet-encdec|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-8B-sae-cc3m_laion-enconly|qwen3-8B-sae-cc3m_laion-enconly|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-8B-sae-cc3m_laion-encdec|qwen3-8B-sae-cc3m_laion-encdec|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-8B-sae-llava_ov-enconly|qwen3-8B-sae-llava_ov-enconly|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-8B-sae-llava_ov-encdec|qwen3-8B-sae-llava_ov-encdec|1|1|${ENV_SAE_LLAVA}"
    # 14B
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-14B-baseline|qwen3-14B-baseline|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-14B-sae-imagenet-enconly|qwen3-14B-sae-imagenet-enconly|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-14B-sae-imagenet-encdec|qwen3-14B-sae-imagenet-encdec|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-14B-sae-cc3m_laion-enconly|qwen3-14B-sae-cc3m_laion-enconly|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-14B-sae-cc3m_laion-encdec|qwen3-14B-sae-cc3m_laion-encdec|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-14B-sae-llava_ov-enconly|qwen3-14B-sae-llava_ov-enconly|1|1|${ENV_SAE_LLAVA}"
    "llava_qwen3|llava_qwen3|$SCRATCH/grob1/llava-more/checkpoints/qwen3-14B-sae-llava_ov-encdec|qwen3-14B-sae-llava_ov-encdec|1|1|${ENV_SAE_LLAVA}"

    # SAE-finetuned LLaVA — needs ENV_SAE_LLAVA (SAE-modified llava package)
    # Simple SAE (encode+decode)
    "sae_llava|llava|fabiangrob/llava-v1.5-7b-finetune-sae|llava-7b-sae-imagenet|1|1|${ENV_SAE_LLAVA}"
    "sae_llava|llava|fabiangrob/llava-v1.5-7b-finetune-sae-cc3m-laion|llava-7b-sae-cc3m-laion|1|1|${ENV_SAE_LLAVA}"
    "sae_llava|llava|fabiangrob/llava-v1.5-13b-finetune-sae|llava-13b-sae-imagenet|1|1|${ENV_SAE_LLAVA}"
    "sae_llava|llava|fabiangrob/llava-v1.5-13b-finetune-sae-cc3m-laion|llava-13b-sae-cc3m-laion|1|1|${ENV_SAE_LLAVA}"
    # Encode-only SAE
    "sae_llava|llava|fabiangrob/llava-v1.5-7b-finetune-sae-encode-only|llava-7b-sae-encode-only-imagenet|1|1|${ENV_SAE_LLAVA}"
    "sae_llava|llava|fabiangrob/llava-v1.5-7b-finetune-sae-encode-only-cc3m-laion|llava-7b-sae-encode-only-cc3m-laion|1|1|${ENV_SAE_LLAVA}"
    "sae_llava|llava|fabiangrob/llava-v1.5-13b-finetune-sae-encode-only|llava-13b-sae-encode-only-imagenet|1|1|${ENV_SAE_LLAVA}"
    "sae_llava|llava|fabiangrob/llava-v1.5-13b-finetune-sae-encode-only-cc3m-laion|llava-13b-sae-encode-only-cc3m-laion|1|1|${ENV_SAE_LLAVA}"
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
# $1: nameref to array of model entries (bash 4.3+)
# $2: time limit
# $3: num_gpus per model
# $4: activate_script path (empty = default main env)
submit_batch() {
    local -n _entries="$1"
    local time="$2" num_gpus="$3" activate_script="$4" pythonpath_extra="${5:-}"

    [ ${#_entries[@]} -eq 0 ] && return

    IFS='|' read -r _ _ _ first_name _ _ _ <<< "${_entries[0]}"
    local job_name="eval-${first_name}"
    [ ${#_entries[@]} -gt 1 ] && job_name="${job_name}+$((${#_entries[@]}-1))"

    local export_vars="ALL,TASKS=${TASKS_FOR_EXPORT},NUM_GPUS=${num_gpus}"
    [ -n "$activate_script" ] && export_vars="${export_vars},ACTIVATE_SCRIPT=${activate_script}"
    [ -n "$pythonpath_extra" ] && export_vars="${export_vars},PYTHONPATH_EXTRA=${pythonpath_extra}"
    for i in "${!_entries[@]}"; do
        local n=$((i+1))
        IFS='|' read -r _ model_type pretrained model_name _ batch_size _ <<< "${_entries[$i]}"
        export_vars="${export_vars},MODEL_TYPE_${n}=${model_type}"
        export_vars="${export_vars},PRETRAINED_${n}=${pretrained}"
        export_vars="${export_vars},MODEL_NAME_${n}=${model_name}"
        export_vars="${export_vars},BATCH_SIZE_${n}=${batch_size}"
    done

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] sbatch --account=$ACCOUNT --nodes=1 --partition=$PARTITION --gres=gpu:4 --time=$time --job-name=$job_name $SLURM_SCRIPT"
        echo "          ACTIVATE_SCRIPT=${activate_script:-<default>}"
        echo "          export: $export_vars"
    else
        echo "Submitting: $job_name (${#_entries[@]} model(s), ${num_gpus}-GPU each, env=${activate_script:-default})"
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

# ---- 1-GPU models: batch up to 4 per node, grouped by activate_script ----
# Collect all unique activate_script values among 1-GPU models.
declare -a UNIQUE_ENVS=()
for entry in "${MODELS[@]}"; do
    IFS='|' read -r family _ _ _ num_gpus _ activate_script <<< "$entry"
    [ -n "$FAMILY_FILTER" ] && [ "$family" != "$FAMILY_FILTER" ] && continue
    [ "$num_gpus" -ne 1 ] && continue
    found=false
    for s in "${UNIQUE_ENVS[@]+"${UNIQUE_ENVS[@]}"}"; do
        [ "$s" = "$activate_script" ] && found=true && break
    done
    $found || UNIQUE_ENVS+=("$activate_script")
done

for current_env in "${UNIQUE_ENVS[@]+"${UNIQUE_ENVS[@]}"}"; do
    SMALL_BATCH=()
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r family _ _ model_name num_gpus _ activate_script <<< "$entry"
        [ -n "$FAMILY_FILTER" ] && [ "$family" != "$FAMILY_FILTER" ] && continue
        [ "$num_gpus" -ne 1 ] && continue
        [ "$activate_script" != "$current_env" ] && continue

        if all_tasks_complete "$RESULTS_DIR/$model_name" "$TASKS"; then
            echo "Skipping: $model_name (all tasks complete)"
            SKIPPED=$((SKIPPED+1))
            continue
        fi

        SMALL_BATCH+=("$entry")
        if [ ${#SMALL_BATCH[@]} -eq 4 ]; then
            local pythonpath_extra=""
            [ "$current_env" = "$ENV_SAE_LLAVA" ] && pythonpath_extra="$REPOS_DIR/LLaVA-MORE"
            submit_batch SMALL_BATCH "$TIME_SMALL" 1 "$current_env" "$pythonpath_extra"
            SUBMITTED_NODES=$((SUBMITTED_NODES+1))
            SMALL_BATCH=()
        fi
    done
    # Flush remainder — still requires a full node even if fewer than 4 models.
    if [ ${#SMALL_BATCH[@]} -gt 0 ]; then
        local pythonpath_extra=""
        [ "$current_env" = "$ENV_SAE_LLAVA" ] && pythonpath_extra="$REPOS_DIR/LLaVA-MORE"
        submit_batch SMALL_BATCH "$TIME_SMALL" 1 "$current_env" "$pythonpath_extra"
        SUBMITTED_NODES=$((SUBMITTED_NODES+1))
    fi
done

# ---- 4-GPU models: one per node ----
for entry in "${MODELS[@]}"; do
    IFS='|' read -r family _ _ model_name num_gpus _ activate_script <<< "$entry"
    [ -n "$FAMILY_FILTER" ] && [ "$family" != "$FAMILY_FILTER" ] && continue
    [ "$num_gpus" -ne 4 ] && continue

    if all_tasks_complete "$RESULTS_DIR/$model_name" "$TASKS"; then
        echo "Skipping: $model_name (all tasks complete)"
        SKIPPED=$((SKIPPED+1))
        continue
    fi

    LARGE_BATCH=("$entry")
    submit_batch LARGE_BATCH "$TIME_LARGE" 4 "$activate_script"
    SUBMITTED_NODES=$((SUBMITTED_NODES+1))
done

echo ""
echo "Submitted $SUBMITTED_NODES node-job(s), skipped $SKIPPED model(s) (already complete)."
