"""VLind-Bench task implementation.

Downloads raw data from klee972/VLind-Bench (~5GB, cached after first run)
and builds the four-stage evaluation pipeline in-memory.
"""

import json
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, DatasetDict

from lmms_eval.api.task import ConfigurableTask


class VLindBenchTask(ConfigurableTask):
    """VLind-Bench: Measuring Language Prior Blindness in Large Vision-Language Models.

    Overrides download() to fetch the raw imagefolder repo from klee972/VLind-Bench
    and expand each instance into per-stage evaluation items.  The expanded dataset
    stores absolute image *paths* (plain strings) — images are opened lazily in
    doc_to_visual(), so the full ~5GB download is cached in HF_HOME and not held in RAM.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        # The task loader passes `class` in the config dict; TaskConfig doesn't
        # know about it, so we remove it before calling super().__init__().
        if config is not None:
            config = {k: v for k, v in config.items() if k != "class"}
        super().__init__(config=config, **kwargs)

    def download(self, dataset_kwargs: Optional[dict] = None) -> None:
        from huggingface_hub import snapshot_download

        repo_id = self.DATASET_PATH or "klee972/VLind-Bench"
        local_dir = Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))

        data_root = local_dir / "VLind-Bench Dataset"
        if not data_root.exists():
            data_root = local_dir

        with open(data_root / "data.json") as f:
            raw_data = json.load(f)

        factual_dir = data_root / "images" / "factual"
        cf_dir = data_root / "images" / "counterfactual"

        rows = self._expand_all(raw_data, factual_dir, cf_dir)
        ds = Dataset.from_list(rows)

        self.dataset = DatasetDict({"train": ds})
        # Remove the image path column for the no-image variant used in logging
        self.dataset_no_image = DatasetDict(
            {"train": ds.remove_columns(["_image_path"])}
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_factual_path(factual_dir: Path, concept: str, context_id: int) -> str:
        d = factual_dir / concept
        if d.exists():
            prefix = f"{context_id}_"
            for folder in d.iterdir():
                if folder.name.startswith(prefix):
                    p = folder / "0.jpg"
                    if p.exists():
                        return str(p)
        return ""

    @staticmethod
    def _find_cf_paths(cf_dir: Path, concept: str, context_id: int) -> list[str]:
        """Return 12 CF image paths (empty string if missing)."""
        d = cf_dir / concept
        if d.exists():
            prefix = f"{context_id}_"
            for folder in d.iterdir():
                if folder.name.startswith(prefix):
                    return [
                        str(p) if (p := folder / f"{i}.jpg").exists() else ""
                        for i in range(12)
                    ]
        return [""] * 12

    def _expand_all(
        self,
        raw_data: list[dict],
        factual_dir: Path,
        cf_dir: Path,
    ) -> list[dict[str, Any]]:
        """Expand 302 instances into per-stage evaluation items.

        Each item carries only the fields needed for its stage plus a single
        _image_path string pointing to the image to use.

        Statement naming in data.json:
          true_statement  = true in CF world (swans in desert)
          false_statement = true in real world (swans in lakes)

        Stage-to-question mapping (q1 always expects "True", q2 "False"):
          CK  q1 = false_statement (factual image — real-world fact IS true)
          CK  q2 = true_statement  (factual image — CF statement is NOT true)
          VP  q1 = existent_noun   (CF image — object IS there)
          VP  q2 = non_existent_noun (CF image — object is NOT there)
          CB  q1 = true_statement  (CF image + context — CF stmt IS true per context)
          CB  q2 = false_statement (CF image + context — factual stmt is NOT true)
          LP  q1 = true_statement  (CF image — CF stmt IS true per image)
          LP  q2 = false_statement (CF image — factual stmt is NOT true per image)
        """
        expanded: list[dict[str, Any]] = []

        for idx, entry in enumerate(raw_data):
            concept = entry["concept"]
            context_id = entry["context_id"]
            best = int(entry.get("best_img_id", 0))

            good_ids = sorted(
                int(k)
                for k, v in entry.get("aggregated_human_label_good_images", {}).items()
                if isinstance(v, (int, float)) and v >= 2
            )

            factual_path = self._find_factual_path(factual_dir, concept, context_id)
            cf_paths = self._find_cf_paths(cf_dir, concept, context_id)
            best_cf_path = cf_paths[best] if best < len(cf_paths) else ""

            base: dict[str, Any] = {
                "instance_id": idx,
                "concept": concept,
                "context": entry["context"],
                "true_statement": entry["true_statement"],
                "false_statement": entry["false_statement"],
                "existent_noun": entry.get("existent_noun", ""),
                "non_existent_noun": entry.get(
                    "non-existent_noun", entry.get("non_existent_noun", "")
                ),
            }

            # CK — factual image
            for qid, stmt_key in [("q1", "false_statement"), ("q2", "true_statement")]:
                expanded.append({
                    **base,
                    "_stage": "ck",
                    "_qid": qid,
                    "_stmt_key": stmt_key,
                    "_noun_key": "",
                    "_cf_img_idx": -1,
                    "_image_path": factual_path,
                })

            # VP — best CF image
            for qid, noun_key in [("q1", "existent_noun"), ("q2", "non_existent_noun")]:
                expanded.append({
                    **base,
                    "_stage": "vp",
                    "_qid": qid,
                    "_stmt_key": "",
                    "_noun_key": noun_key,
                    "_cf_img_idx": best,
                    "_image_path": best_cf_path,
                })

            # CB — best CF image
            for qid, stmt_key in [("q1", "true_statement"), ("q2", "false_statement")]:
                expanded.append({
                    **base,
                    "_stage": "cb",
                    "_qid": qid,
                    "_stmt_key": stmt_key,
                    "_noun_key": "",
                    "_cf_img_idx": best,
                    "_image_path": best_cf_path,
                })

            # LP — one item per good CF image
            for cf_idx in good_ids:
                cf_path = cf_paths[cf_idx] if cf_idx < len(cf_paths) else ""
                for qid, stmt_key in [("q1", "true_statement"), ("q2", "false_statement")]:
                    expanded.append({
                        **base,
                        "_stage": "lp",
                        "_qid": qid,
                        "_stmt_key": stmt_key,
                        "_noun_key": "",
                        "_cf_img_idx": cf_idx,
                        "_image_path": cf_path,
                    })

        return expanded
