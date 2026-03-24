"""Prepare VLind-Bench dataset for lmms-eval.

Downloads the raw klee972/VLind-Bench repo (imagefolder format),
parses data.json and uploads a proper tabular dataset to fabiangrob/vlind-bench.

CF images are stored as individual columns cf_image_0 ... cf_image_11 to avoid
a datasets/PyArrow bug with Sequence(Image()).

Usage:
    git clone https://huggingface.co/datasets/klee972/VLind-Bench /tmp/vlind_bench_git
    cd /tmp/vlind_bench_git && git lfs pull
    python scripts/prepare_vlind_bench.py

Requires HF login (huggingface-cli login) before running.
"""

import json
import tempfile
from pathlib import Path
from typing import Optional

from datasets import Dataset, Features, Image, Sequence, Value
from huggingface_hub import snapshot_download
from PIL import Image as PILImage


def make_placeholder_path() -> str:
    """Create a 1×1 white JPEG placeholder for missing images."""
    p = Path(tempfile.mktemp(suffix=".jpg"))
    PILImage.new("RGB", (1, 1), color=(255, 255, 255)).save(p, "JPEG")
    return str(p)


def find_factual_image_path(factual_dir: Path, concept: str, context_id: int) -> Optional[str]:
    concept_dir = factual_dir / concept
    if not concept_dir.exists():
        return None
    prefix = f"{context_id}_"
    for folder in concept_dir.iterdir():
        if folder.name.startswith(prefix):
            p = folder / "0.jpg"
            return str(p) if p.exists() else None
    return None


def find_cf_image_paths(cf_dir: Path, concept: str, context_id: int, placeholder: str) -> list[str]:
    """Return list of 12 CF image paths (placeholder if missing)."""
    concept_dir = cf_dir / concept
    if not concept_dir.exists():
        return [placeholder] * 12
    prefix = f"{context_id}_"
    for folder in concept_dir.iterdir():
        if folder.name.startswith(prefix):
            return [
                str(folder / f"{i}.jpg") if (folder / f"{i}.jpg").exists() else placeholder
                for i in range(12)
            ]
    return [placeholder] * 12


def main() -> None:
    placeholder = make_placeholder_path()

    git_clone = Path("/tmp/vlind_bench_git")
    if git_clone.exists():
        print("Using existing git clone at /tmp/vlind_bench_git")
        local_dir = git_clone
    else:
        print("Downloading klee972/VLind-Bench from HuggingFace...")
        local_dir = Path(snapshot_download(
            repo_id="klee972/VLind-Bench",
            repo_type="dataset",
            local_dir="/tmp/vlind_bench_raw",
        ))

    data_root = local_dir / "VLind-Bench Dataset"
    if not data_root.exists():
        data_root = local_dir
    assert data_root.exists(), f"Data root not found: {data_root}"

    data_json_path = data_root / "data.json"
    assert data_json_path.exists(), f"data.json not found: {data_json_path}"

    factual_dir = data_root / "images" / "factual"
    cf_dir = data_root / "images" / "counterfactual"

    print(f"Parsing {data_json_path}...")
    with open(data_json_path) as f:
        raw_data = json.load(f)
    print(f"Found {len(raw_data)} instances in data.json")

    # Column-oriented dict. CF images are flattened into 12 separate columns
    # to avoid the datasets Sequence(Image()) / PyArrow 22 bug.
    cols: dict = {
        "instance_id": [], "global_id": [], "concept": [],
        "context": [], "true_statement": [], "false_statement": [],
        "existent_noun": [], "non_existent_noun": [],
        "best_img_id": [], "good_img_ids": [],
        "factual_image": [],
        **{f"cf_image_{i}": [] for i in range(12)},
    }

    missing_factual = 0
    missing_cf = 0

    for idx, entry in enumerate(raw_data):
        if idx % 50 == 0:
            print(f"Processing {idx}/{len(raw_data)}...")

        concept = entry["concept"]
        context_id = entry["context_id"]

        label_votes = entry.get("aggregated_human_label_good_images", {})
        good_img_ids = sorted(
            int(img_id)
            for img_id, votes in label_votes.items()
            if isinstance(votes, (int, float)) and votes >= 2
        )

        factual_path = find_factual_image_path(factual_dir, concept, context_id)
        if factual_path is None:
            missing_factual += 1
        cf_paths = find_cf_image_paths(cf_dir, concept, context_id, placeholder)
        if all(p == placeholder for p in cf_paths):
            missing_cf += 1

        cols["instance_id"].append(idx)
        cols["global_id"].append(entry.get("global_id", idx))
        cols["concept"].append(concept)
        cols["context"].append(entry["context"])
        cols["true_statement"].append(entry["true_statement"])
        cols["false_statement"].append(entry["false_statement"])
        cols["existent_noun"].append(entry.get("existent_noun", ""))
        cols["non_existent_noun"].append(
            entry.get("non-existent_noun", entry.get("non_existent_noun", ""))
        )
        cols["best_img_id"].append(int(entry.get("best_img_id", 0)))
        cols["good_img_ids"].append(good_img_ids)
        cols["factual_image"].append(factual_path if factual_path else placeholder)
        for i, p in enumerate(cf_paths):
            cols[f"cf_image_{i}"].append(p)

    print(f"\n{len(raw_data)} instances | {missing_factual} missing factual | {missing_cf} missing CF")

    features = Features({
        "instance_id": Value("int32"),
        "global_id": Value("int32"),
        "concept": Value("string"),
        "context": Value("string"),
        "true_statement": Value("string"),
        "false_statement": Value("string"),
        "existent_noun": Value("string"),
        "non_existent_noun": Value("string"),
        "best_img_id": Value("int32"),
        "good_img_ids": Sequence(Value("int32")),
        "factual_image": Image(),
        **{f"cf_image_{i}": Image() for i in range(12)},
    })

    print("\nBuilding HuggingFace Dataset...")
    dataset = Dataset.from_dict(cols, features=features)
    print(f"Dataset: {dataset}")

    # Sanity check
    row = dataset[0]
    print(f"\nSample row 0: concept={row['concept']}, best_img_id={row['best_img_id']}, "
          f"good_img_ids={row['good_img_ids']}")
    print(f"  factual_image: {type(row['factual_image']).__name__} {row['factual_image'].size}")
    best = row['best_img_id']
    print(f"  cf_image_{best}: {type(row[f'cf_image_{best}']).__name__} {row[f'cf_image_{best}'].size}")

    print("\nPushing to fabiangrob/vlind-bench...")
    dataset.push_to_hub(
        "fabiangrob/vlind-bench",
        commit_message="Add restructured VLind-Bench dataset for lmms-eval (source: klee972/VLind-Bench)",
    )
    print("Done! https://huggingface.co/datasets/fabiangrob/vlind-bench")


if __name__ == "__main__":
    main()
