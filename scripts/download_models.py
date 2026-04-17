#!/usr/bin/env python3
"""
Download all lmms-eval model checkpoints to HF_HOME on the login node.
Compute nodes have no internet access, so this must be run first.

Usage:
    source sc_venv_template/activate.sh
    python scripts/download_models.py               # all models
    python scripts/download_models.py --dry-run     # print what would be downloaded
    python scripts/download_models.py internvl3     # filter by family name
"""

import argparse
import os
import sys

MODELS = [
    # family, repo_id
    # InternVL3.5
    ("internvl3_5", "OpenGVLab/InternVL3_5-1B"),
    ("internvl3_5", "OpenGVLab/InternVL3_5-2B"),
    ("internvl3_5", "OpenGVLab/InternVL3_5-4B"),
    ("internvl3_5", "OpenGVLab/InternVL3_5-8B"),
    ("internvl3_5", "OpenGVLab/InternVL3_5-14B"),
    ("internvl3_5", "OpenGVLab/InternVL3_5-38B"),
    # InternVL3
    ("internvl3", "OpenGVLab/InternVL3-1B"),
    ("internvl3", "OpenGVLab/InternVL3-2B"),
    ("internvl3", "OpenGVLab/InternVL3-8B"),
    ("internvl3", "OpenGVLab/InternVL3-9B"),
    ("internvl3", "OpenGVLab/InternVL3-14B"),
    ("internvl3", "OpenGVLab/InternVL3-38B"),
    ("internvl3", "OpenGVLab/InternVL3-78B"),
    # Qwen2.5-VL
    ("qwen25vl", "Qwen/Qwen2.5-VL-3B-Instruct"),
    ("qwen25vl", "Qwen/Qwen2.5-VL-7B-Instruct"),
    ("qwen25vl", "Qwen/Qwen2.5-VL-32B-Instruct"),
    ("qwen25vl", "Qwen/Qwen2.5-VL-72B-Instruct"),
    # Qwen3-VL
    ("qwen3vl", "Qwen/Qwen3-VL-2B-Instruct"),
    ("qwen3vl", "Qwen/Qwen3-VL-4B-Instruct"),
    ("qwen3vl", "Qwen/Qwen3-VL-8B-Instruct"),
    ("qwen3vl", "Qwen/Qwen3-VL-32B-Instruct"),
    # LLaVA-OneVision
    ("llava_ov", "lmms-lab/llava-onevision-qwen2-0.5b-ov"),
    ("llava_ov", "lmms-lab/llava-onevision-qwen2-7b-ov"),
    ("llava_ov", "lmms-lab/llava-onevision-qwen2-72b-ov"),
    # Gemma-3
    ("gemma3", "google/gemma-3-4b-it"),
    ("gemma3", "google/gemma-3-12b-it"),
    ("gemma3", "google/gemma-3-27b-it"),
    # LLaVA 1.5
    ("llava15", "liuhaotian/llava-v1.5-7b"),
    ("llava15", "liuhaotian/llava-v1.5-13b"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", nargs="?", help="Filter to a specific model family")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        print("ERROR: HF_HOME is not set. Add it to ~/.bashrc and re-login.", file=sys.stderr)
        sys.exit(1)

    from huggingface_hub import snapshot_download

    models = [(f, r) for f, r in MODELS if not args.family or f == args.family]
    if not models:
        print(f"No models found for family '{args.family}'.")
        sys.exit(1)

    print(f"HF_HOME: {hf_home}")
    print(f"Downloading {len(models)} model(s):\n")

    failed = []
    for i, (family, repo_id) in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {repo_id}")
        if args.dry_run:
            continue
        try:
            path = snapshot_download(repo_id=repo_id, resume_download=True)
            print(f"  -> {path}")
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(repo_id)

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for r in failed:
            print(f"  {r}")
        sys.exit(1)
    elif not args.dry_run:
        print(f"\nAll {len(models)} model(s) downloaded successfully.")


if __name__ == "__main__":
    main()
