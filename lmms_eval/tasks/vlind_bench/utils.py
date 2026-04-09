"""Utility functions for VLind-Bench: Measuring Language Prior Blindness in LVLMs.

Four-stage pipeline benchmark:
  CK  (Commonsense Knowledge)  – factual image, tests real-world knowledge
  VP  (Visual Perception)      – CF image, tests object recognition
  CB  (Commonsense Bias)       – CF image + text context, tests with textual scaffolding
  LP  (Language Prior)         – CF image only, the core LP measurement

Each stage asks two binary True/False questions. q1 always expects "True", q2 "False".
A stage passes only when both questions are answered correctly.

Pipeline: S_LP is only counted when CK + VP + CB all pass for that instance.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# doc_to_visual / doc_to_text / doc_to_target
# ---------------------------------------------------------------------------

def vlind_bench_doc_to_visual(doc: dict[str, Any]) -> list:
    """Open and return the image for this evaluation item.

    Images are stored as absolute paths in _image_path; missing images return [].
    Models resize internally so we pass images at their original resolution.
    """
    path = doc.get("_image_path", "")
    if not path or not Path(path).exists():
        return []
    return [PILImage.open(path).convert("RGB")]


def vlind_bench_doc_to_text(
    doc: dict[str, Any],
    lmms_eval_specific_kwargs: Optional[dict[str, str]] = None,
) -> str:
    """Build the prompt for this evaluation item."""
    stage = doc["_stage"]

    if stage == "ck":
        stmt = doc[doc["_stmt_key"]]
        return (
            f"Statement: {stmt}\n"
            "Based on common sense, is the given statement true or false?\n"
            "Only respond in True or False."
        )

    if stage == "vp":
        noun = doc[doc["_noun_key"]]
        return (
            f"Statement: There is {noun} in the given image.\n"
            "Based on the image, is the given statement true or false?\n"
            "Only respond in True or False."
        )

    if stage == "cb":
        stmt = doc[doc["_stmt_key"]]
        context = doc["context"]
        return (
            f"Context: {context}\n"
            f"Statement: {stmt}\n"
            "Based on the context, is the given statement true or false? "
            "Forget real-world common sense and just follow the information provided in the context.\n"
            "Only respond in True or False."
        )

    # LP
    stmt = doc[doc["_stmt_key"]]
    return (
        f"Statement: {stmt}\n"
        "Based on the image, is the given statement true or false? "
        "Forget real-world common sense and just follow the information provided in the image.\n"
        "Only respond in True or False."
    )


def vlind_bench_doc_to_target(doc: dict[str, Any]) -> str:
    """q1 always expects 'True', q2 always expects 'False'."""
    return "True" if doc["_qid"] == "q1" else "False"


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _extract_true_false(text: str) -> Optional[str]:
    """Left-to-right word scan for 'true'/'false' (matches original infer_true_or_false)."""
    for word in text.lower().replace("\n", " ").replace(",", "").replace(".", "").split():
        if word == "true":
            return "True"
        if word == "false":
            return "False"
    return None


# ---------------------------------------------------------------------------
# process_results
# ---------------------------------------------------------------------------

def vlind_bench_process_results(
    doc: dict[str, Any], results: list[str]
) -> dict[str, Any]:
    """Score one evaluation item and return a tagged result under every metric key."""
    pred = _extract_true_false(results[0].strip())
    target = vlind_bench_doc_to_target(doc)
    correct = pred == target if pred is not None else False

    result_info = {
        "instance_id": doc["instance_id"],
        "stage": doc["_stage"],
        "qid": doc["_qid"],
        "cf_img_idx": doc["_cf_img_idx"],
        "correct": correct,
        "prediction": pred,
        "target": target,
        "raw_prediction": results[0].strip(),
        "concept": doc.get("concept", ""),
    }

    # Return the same dict under every metric key — each aggregation function
    # filters to its relevant stage and applies the pipeline logic.
    return {
        "s_ck": result_info,
        "s_vp": result_info,
        "s_cb": result_info,
        "s_lp": result_info,
        "accuracy_cb": result_info,
        "accuracy_lp": result_info,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _group_by_instance(results: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list] = defaultdict(list)
    for r in results:
        groups[r["instance_id"]].append(r)
    return groups


def _stage_passes(items: list[dict[str, Any]], stage: str) -> bool:
    """True iff both q1 and q2 are correct for this (non-LP) stage."""
    stage_items = [r for r in items if r["stage"] == stage]
    q1_ok = any(r["correct"] for r in stage_items if r["qid"] == "q1")
    q2_ok = any(r["correct"] for r in stage_items if r["qid"] == "q2")
    return q1_ok and q2_ok


def _lp_image_passes(items: list[dict[str, Any]]) -> list[bool]:
    """Per-image LP pass list (pass = both q1 and q2 correct for that image)."""
    by_img: dict[int, list] = defaultdict(list)
    for r in items:
        if r["stage"] == "lp":
            by_img[r["cf_img_idx"]].append(r)
    return [
        any(r["correct"] for r in img_items if r["qid"] == "q1")
        and any(r["correct"] for r in img_items if r["qid"] == "q2")
        for img_items in by_img.values()
    ]


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------

def aggregate_s_ck(results: list[dict[str, Any]]) -> float:
    """CK pass ratio over all instances."""
    groups = _group_by_instance(results)
    passes = [_stage_passes(items, "ck") for items in groups.values()]
    return sum(passes) / len(passes) if passes else 0.0


def aggregate_s_vp(results: list[dict[str, Any]]) -> float:
    """VP pass ratio over all instances."""
    groups = _group_by_instance(results)
    passes = [_stage_passes(items, "vp") for items in groups.values()]
    return sum(passes) / len(passes) if passes else 0.0


def aggregate_s_cb(results: list[dict[str, Any]]) -> float:
    """CB pass ratio — denominator is instances where CK passes."""
    groups = _group_by_instance(results)
    num, den = 0, 0
    for items in groups.values():
        if _stage_passes(items, "ck"):
            den += 1
            if _stage_passes(items, "cb"):
                num += 1
    return num / den if den > 0 else 0.0


def aggregate_s_lp(results: list[dict[str, Any]]) -> float:
    """LP pass ratio (macro) — only instances where CK + VP + CB all pass.

    For each qualifying instance: fraction of good CF images where both
    LP questions are correct. Then average over qualifying instances.
    """
    groups = _group_by_instance(results)
    instance_scores = []
    for items in groups.values():
        if _stage_passes(items, "ck") and _stage_passes(items, "vp") and _stage_passes(items, "cb"):
            img_passes = _lp_image_passes(items)
            if img_passes:
                instance_scores.append(sum(img_passes) / len(img_passes))
    return sum(instance_scores) / len(instance_scores) if instance_scores else 0.0


def aggregate_accuracy_cb(results: list[dict[str, Any]]) -> float:
    """Raw CB item-level accuracy, no pipeline filtering."""
    cb_items = [r for r in results if r["stage"] == "cb"]
    if not cb_items:
        return 0.0
    return sum(r["correct"] for r in cb_items) / len(cb_items)


def aggregate_accuracy_lp(results: list[dict[str, Any]]) -> float:
    """Raw LP item-level accuracy, no pipeline filtering."""
    lp_items = [r for r in results if r["stage"] == "lp"]
    if not lp_items:
        return 0.0
    return sum(r["correct"] for r in lp_items) / len(lp_items)
