"""Patch existing vlind_bench result files to add accuracy_cb metric.

accuracy_cb was added to utils.py after the initial eval runs, so older
_results.json files are missing it. This script recomputes it from the
per-sample .jsonl file and injects the value into _results.json in-place.

Usage:
    # Patch all results under the default results directory
    python scripts/patch_vlind_accuracy_cb.py

    # Patch a specific results root
    python scripts/patch_vlind_accuracy_cb.py --results-dir results/sae_llava_models_13b
"""

import argparse
import json
from pathlib import Path


def compute_accuracy_cb(samples_path: Path) -> float:
    """Recompute raw CB item-level accuracy from a samples .jsonl file."""
    cb_items = []
    with open(samples_path) as f:
        for line in f:
            doc = json.loads(line)
            s_cb = doc.get("s_cb")
            if s_cb and s_cb.get("stage") == "cb":
                cb_items.append(s_cb["correct"])
    if not cb_items:
        return 0.0
    return sum(cb_items) / len(cb_items)


def patch_results_file(results_path: Path, accuracy_cb: float) -> None:
    """Inject accuracy_cb into a _results.json file if not already present."""
    with open(results_path) as f:
        data = json.load(f)

    vl = data.get("results", {}).get("vlind_bench")
    if vl is None:
        return  # no vlind_bench results in this file

    if "accuracy_cb,none" in vl:
        print(f"  already patched, skipping: {results_path}")
        return

    # Insert accuracy_cb right after accuracy_lp (or at end of vlind_bench block)
    # We rebuild the dict to control key ordering
    new_vl = {}
    for k, v in vl.items():
        new_vl[k] = v
        if k == "s_cb,none":
            # Insert accuracy_cb right after s_cb (mirrors accuracy_lp after s_lp)
            new_vl["accuracy_cb,none"] = accuracy_cb
            new_vl["accuracy_cb_stderr,none"] = "N/A"
            new_vl["accuracy_cb_stderr_clt,none"] = "N/A"
            new_vl["accuracy_cb_stderr_clustered,none"] = "N/A"

    data["results"]["vlind_bench"] = new_vl

    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  patched: {results_path.parent.name} — accuracy_cb = {accuracy_cb:.4f}")


def patch_directory(results_root: Path) -> None:
    """Walk a results directory and patch all vlind_bench result files."""
    results_files = sorted(results_root.rglob("*_results.json"))
    if not results_files:
        print(f"No *_results.json files found under {results_root}")
        return

    patched = 0
    for results_path in results_files:
        run_dir = results_path.parent
        samples_path = next(run_dir.glob("*_samples_vlind_bench.jsonl"), None)
        if samples_path is None:
            print(f"  no samples file found, skipping: {run_dir}")
            continue

        accuracy_cb = compute_accuracy_cb(samples_path)
        patch_results_file(results_path, accuracy_cb)
        patched += 1

    print(f"\nDone. Patched {patched} file(s) under {results_root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root directory to search for *_results.json files (default: results/)",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Results directory not found: {args.results_dir}")
        return

    patch_directory(args.results_dir)


if __name__ == "__main__":
    main()
