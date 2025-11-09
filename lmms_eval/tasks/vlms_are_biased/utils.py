"""Utility functions for VLMs Are Biased benchmark."""

from collections import defaultdict
from typing import Any


def vlms_are_biased_doc_to_visual(doc: dict[str, Any]) -> list:
    """Extract image from document.

    Args:
        doc: Document containing image field

    Returns:
        List containing the RGB image
    """
    return [doc["image"].convert("RGB")]


def vlms_are_biased_doc_to_text(
    doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, str] | None = None
) -> str:
    """Format question text with optional prompt additions.

    Args:
        doc: Document containing question or prompt field
        lmms_eval_specific_kwargs: Optional pre/post prompts

    Returns:
        Formatted question string
    """
    # Try different field names that might contain the question
    question = doc.get("question", doc.get("prompt", doc.get("text", "")))

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    return f"{pre_prompt}{question}{post_prompt}"


def vlms_are_biased_process_results(
    doc: dict[str, Any], results: list[str]
) -> dict[str, Any]:
    """Process model results and compute accuracy.

    Args:
        doc: Document containing ground truth answer
        results: List containing model prediction

    Returns:
        Dictionary with accuracy metric and metadata
    """
    pred = results[0].strip().lower()
    answer = str(doc["answer"]).strip().lower()

    # Check for exact match
    is_correct = pred == answer

    # Also check if the answer is contained in or contains the prediction
    # This handles cases like "4" vs "four" or "4 stripes"
    if not is_correct:
        # Extract numbers from both strings
        pred_numbers = "".join(c for c in pred if c.isdigit())
        answer_numbers = "".join(c for c in answer if c.isdigit())
        if pred_numbers and answer_numbers:
            is_correct = pred_numbers == answer_numbers

    domain = doc.get("domain", doc.get("category", "unknown"))

    return {
        "accuracy": float(is_correct),
        "accuracy_by_domain": {"domain": domain, "correct": is_correct},
    }


def vlms_are_biased_aggregate_by_domain(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    """Aggregate results by domain.

    Args:
        results: List of result dictionaries with domain and correctness

    Returns:
        Dictionary mapping domain names to accuracy scores
    """
    domain_correct: dict[str, int] = defaultdict(int)
    domain_total: dict[str, int] = defaultdict(int)

    for result in results:
        domain = result["domain"]
        correct = result["correct"]

        domain_total[domain] += 1
        if correct:
            domain_correct[domain] += 1

    # Calculate accuracy per domain
    domain_accuracy = {}
    for domain in domain_total:
        accuracy = domain_correct[domain] / domain_total[domain]
        domain_accuracy[domain] = accuracy

    # Add overall accuracy
    total_correct = sum(domain_correct.values())
    total = sum(domain_total.values())
    domain_accuracy["overall"] = total_correct / total if total > 0 else 0.0

    return domain_accuracy
