# Reward function for V*Bench / OPD (multiple-choice visual QA with <answer> tags)

import re
from typing import Any, Optional, Tuple


# Metadata
REWARD_NAME = "vstar"
REWARD_TYPE = "batch"


def _extract_answer_content(text: str) -> Tuple[str, bool]:
    """Extract content from <answer>...</answer> tags."""
    if text is None:
        return "", False
    match = re.search(r"<answer>\s*(.*?)\s*(?:</answer>|$)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), True
    return text.strip(), False


def _normalize_answer_text(text: str) -> str:
    """Normalize answer text for matching."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.upper()


def _normalize_exact_choice(answer_content: str) -> Optional[str]:
    """Try to extract a single choice letter (A-D) from answer content."""
    text = _normalize_answer_text(answer_content)
    if not text:
        return None

    candidates = [
        re.fullmatch(r"\\BOXED\{\s*([A-D])\s*\}", text),
        re.fullmatch(r"([A-D])", text),
        re.fullmatch(r"\(?\s*([A-D])\s*\)?\.?", text),
        re.match(r"^\(?\s*([A-D])\s*\)?(?:[\.\:\-\s]+|$)", text),
        re.search(r"\b(?:OPTION)\s*([A-D])\b", text),
        re.search(r"\b(?:ANSWER)\s*(?:IS|:)?\s*([A-D])\b", text),
    ]
    for match in candidates:
        if match:
            return match.group(1).upper()
    return None


def extract_answer_letter(response: str) -> Optional[str]:
    """Extract the answer letter from a model response."""
    # First try <answer>...</answer> tag
    content, found_tag = _extract_answer_content(response)
    if found_tag:
        letter = _normalize_exact_choice(content)
        if letter:
            return letter

    # Fallback: try to extract from the full response
    return _normalize_exact_choice(response)


def format_reward(response: str) -> float:
    """Check if the response ends with <answer>...</answer>."""
    _, found = _extract_answer_content(response)
    return 1.0 if found else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Check if the extracted answer matches the ground truth letter."""
    predicted = extract_answer_letter(response)
    if predicted is None:
        return 0.0
    return 1.0 if predicted == ground_truth.strip().upper() else 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )
    return scores
