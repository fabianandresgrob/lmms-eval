"""Utility functions for vlind-bench-oe evaluation.

Open-ended version of VLind-Bench: one row per (instance × good CF image).
The model receives a question + instruction suffix and must produce a short
answer that is scored against expected_answers (correct) or biased_answers
(biased).

Scoring is substring-based after normalization (lowercase, strip articles and
punctuation) — both directions checked:
  - model answer contained in a reference answer  ("desert" in "desert sands")
  - reference answer contained in model answer    ("desert" in "sandy desert area")

NOTE: This file intentionally duplicates the scoring logic from
vlm-mechanistic-analysis/dataset_creation/vlind_bench_oe_utils.py to avoid
cross-repo imports in lmms-eval.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Scoring (duplicated from vlind_bench_oe_utils.py)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip articles and punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _score_response(
    response: str,
    expected_answers: list[str],
    biased_answers: list[str],
) -> str:
    """Return 'correct', 'biased', or 'other'."""
    resp = _normalize(response)
    if not resp:
        return "other"

    for ans in expected_answers:
        a = _normalize(ans)
        if a and (a in resp or resp in a):
            return "correct"

    for ans in biased_answers:
        a = _normalize(ans)
        if a and (a in resp or resp in a):
            return "biased"

    return "other"


# ---------------------------------------------------------------------------
# doc_to_visual / doc_to_text / doc_to_target
# ---------------------------------------------------------------------------

def vlind_bench_oe_doc_to_visual(doc: dict[str, Any]) -> list:
    """Return the CF image for this sample.

    The dataset stores images as PIL Images in the 'cf_image' column.
    """
    img = doc.get("cf_image")
    if img is None:
        return []
    if isinstance(img, PILImage.Image):
        return [img.convert("RGB")]
    # Fallback: path string
    path = str(img)
    if Path(path).exists():
        return [PILImage.open(path).convert("RGB")]
    return []


def vlind_bench_oe_doc_to_text(
    doc: dict[str, Any],
    lmms_eval_specific_kwargs: Optional[dict[str, str]] = None,
) -> str:
    """Return question + instructions."""
    instructions = doc.get("instructions", "\nAnswer the question using a single word or phrase.")
    return doc["question"] + instructions


def vlind_bench_oe_doc_to_target(doc: dict[str, Any]) -> str:
    """Return the first expected answer as the primary target."""
    answers = doc.get("expected_answers", [])
    return answers[0] if answers else ""


# ---------------------------------------------------------------------------
# process_results
# ---------------------------------------------------------------------------

def vlind_bench_oe_process_results(
    doc: dict[str, Any], results: list[str]
) -> dict[str, Any]:
    """Score one sample and return result dict under every metric key."""
    response = results[0].strip() if results else ""
    score = _score_response(
        response,
        doc.get("expected_answers", []),
        doc.get("biased_answers", []),
    )

    result_info = {
        "instance_id":      doc["instance_id"],
        "cf_img_idx":       doc["cf_img_idx"],
        "concept":          doc.get("concept", ""),
        "score":            score,
        "correct":          score == "correct",
        "biased":           score == "biased",
        "response":         response,
        "expected_answers": doc.get("expected_answers", []),
        "biased_answers":   doc.get("biased_answers", []),
    }

    return {
        "accuracy":   result_info,
        "bias_rate":  result_info,
        "other_rate": result_info,
    }


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------

def aggregate_accuracy(results: list[dict[str, Any]]) -> float:
    """Fraction of samples scored as 'correct'."""
    if not results:
        return 0.0
    return sum(r["correct"] for r in results) / len(results)


def aggregate_bias_rate(results: list[dict[str, Any]]) -> float:
    """Fraction of samples scored as 'biased'."""
    if not results:
        return 0.0
    return sum(r["biased"] for r in results) / len(results)


def aggregate_other_rate(results: list[dict[str, Any]]) -> float:
    """Fraction of samples scored as 'other' (neither correct nor biased)."""
    if not results:
        return 0.0
    return sum(r["score"] == "other" for r in results) / len(results)
