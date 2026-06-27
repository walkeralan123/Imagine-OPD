# Reward function for TreeVGR free-form textual answers with question-type-aware
# canonical label extraction and exact label matching.

import re
from typing import Any, Callable, Optional, Tuple


REWARD_NAME = "treevgr_text"
REWARD_TYPE = "batch"

_COLOR_ALIASES = {
    "grey": "gray",
    "gray": "gray",
    "silver": "silver",
    "golden": "gold",
}
_COLORS = {
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "brown",
    "tan",
    "gray",
    "grey",
    "orange",
    "pink",
    "purple",
    "silver",
    "gold",
    "beige",
    "blonde",
    "bronze",
    "maroon",
    "turquoise",
    "violet",
}
_MATERIAL_ALIASES = {
    "wooden": "wood",
    "metallic": "metal",
}
_MATERIALS = {
    "wood",
    "wooden",
    "plastic",
    "metal",
    "chrome",
    "glass",
    "ceramic",
    "brick",
    "stone",
    "concrete",
    "paper",
    "cardboard",
    "leather",
    "fabric",
    "cloth",
    "rubber",
    "steel",
    "metallic",
}
_SPATIAL_PATTERNS = [
    (r"\bin front of\b", "front"),
    (r"\bbehind\b", "behind"),
    (r"\bto the left\b", "left"),
    (r"\bleft side\b", "left"),
    (r"\bto the right\b", "right"),
    (r"\bright side\b", "right"),
    (r"\babove\b", "above"),
    (r"\bbelow\b", "below"),
    (r"\bupper\b", "top"),
    (r"\btop\b", "top"),
    (r"\blower\b", "bottom"),
    (r"\bbottom\b", "bottom"),
    (r"\bmiddle\b", "middle"),
    (r"\bcenter\b", "middle"),
    (r"\bcentre\b", "middle"),
    (r"\bleft\b", "left"),
    (r"\bright\b", "right"),
]
_ATTRIBUTE_ALIASES = {
    "wrinkly": "wrinkled",
    "circular": "round",
    "smile": "smiling",
    "laugh": "laughing",
}
_ATTRIBUTE_WORDS = {
    "serious",
    "happy",
    "sad",
    "smiling",
    "laughing",
    "grinning",
    "sleepy",
    "surprised",
    "angry",
    "wet",
    "dry",
    "full",
    "empty",
    "open",
    "closed",
    "striped",
    "plaid",
    "checkered",
    "plain",
    "round",
    "square",
    "rectangular",
    "wrinkled",
    "smooth",
    "large",
    "small",
    "tall",
    "short",
    "clean",
    "dirty",
    "jumping",
    "standing",
    "sitting",
    "walking",
    "running",
    "squatting",
}
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_QUESTION_PATTERNS_SUBJECT = (
    "what is on ",
    "what is in ",
    "what is under ",
    "what is beside ",
    "what is next to ",
    "what animal ",
    "what vegetable ",
    "what fruit ",
    "what else ",
    "what can be seen",
)
_QUESTION_PATTERNS_OBJECT = (
    " is on what",
    " is in what",
    " is under what",
    " is beside what",
    " is next to what",
)
_ARTICLES = ("a ", "an ", "the ", "some ")
_FILLER_PREFIXES = (
    "indeed ",
    "actually ",
    "visible as ",
    "visible ",
    "located ",
    "positioned ",
    "situated ",
)


def _extract_answer_content(text: str) -> Tuple[str, bool]:
    if text is None:
        return "", False
    match = re.search(r"<answer>\s*(.*?)\s*(?:</answer>|$)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), True
    return text.strip(), False


def _normalize_question(text: str) -> str:
    text = (text or "").replace("<image>", " ").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_common_prefixes(text: str) -> str:
    text = (text or "").strip()
    prefixes = (
        r"^(?:final answer|answer)\s*[:\-]\s*",
        r"^(?:the answer is|it is|it's)\s+",
    )
    for pattern in prefixes:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()


def _normalize_text(text: str) -> str:
    text = _strip_common_prefixes(text.lower())
    text = re.sub(r"</?answer>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9\s,]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,.")


def _strip_articles(text: str) -> str:
    for article in _ARTICLES:
        if text.startswith(article):
            return text[len(article) :].strip()
    return text.strip()


def _strip_yes_no_prefix(text: str) -> str:
    text = _normalize_text(text)
    return re.sub(r"^(?:yes|no)\b[\s,]*", "", text).strip()


def _canonical_phrase(text: str) -> str:
    text = _normalize_text(text)
    text = _strip_articles(text)
    text = re.sub(r"\b(?:appears|looks|seems)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,.")


def _normalize_list_phrase(text: str) -> str:
    text = _canonical_phrase(text)
    if "," not in text and " and " not in text:
        return text
    parts = re.split(r",| and ", text)
    parts = [_strip_articles(part.strip()) for part in parts if part.strip()]
    return " | ".join(sorted(dict.fromkeys(parts)))


def _extract_yes_no_label(text: str) -> Optional[str]:
    match = re.match(r"^(yes|no)\b", _normalize_text(text))
    return match.group(1) if match else None


def _extract_vocab_label(text: str, vocab: set[str], aliases: dict[str, str]) -> Optional[str]:
    normalized = _normalize_text(text)
    for token in normalized.split():
        if token in aliases:
            return aliases[token]
        if token in vocab:
            return token
    return None


def _extract_spatial_label(text: str) -> Optional[str]:
    normalized = _normalize_text(text)
    for pattern, label in _SPATIAL_PATTERNS:
        if re.search(pattern, normalized):
            return label
    return None


def _extract_number_label(text: str) -> Optional[str]:
    normalized = _normalize_text(text)
    digit = re.search(r"\b\d+\b", normalized)
    if digit:
        return digit.group(0)
    for token in normalized.split():
        if token in _NUMBER_WORDS:
            return str(_NUMBER_WORDS[token])
    return None


def _extract_attribute_label(text: str) -> Optional[str]:
    normalized = _strip_yes_no_prefix(text)
    for token in normalized.split():
        if token in _ATTRIBUTE_ALIASES:
            return _ATTRIBUTE_ALIASES[token]
        if token in _ATTRIBUTE_WORDS:
            return token
    return None


def _extract_predicate_phrase(text: str) -> str:
    normalized = _strip_yes_no_prefix(text)
    for filler in _FILLER_PREFIXES:
        normalized = normalized.replace(filler, "")
    patterns = (
        r"\b(?:is|are|was|were)\s+(.+)$",
        r"\b(?:wearing|wears|holding|holds|riding|rides|standing|sitting|walking|running)\s+(.+)$",
        r"\b(?:on|in|under|beside|behind|above|below)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return _normalize_list_phrase(match.group(1))
    return _normalize_list_phrase(normalized)


def _extract_subject_phrase(text: str) -> str:
    normalized = _strip_yes_no_prefix(text)
    match = re.match(
        r"^(?:a|an|the|some)\s+(.+?)\s+(?:is|are|was|were|wears|wear|walks|walk|holds|hold|rides|ride|stands|stand|sits|sit)\b",
        normalized,
    )
    if match:
        return _canonical_phrase(match.group(1))
    return _canonical_phrase(normalized)


def _extract_object_after_preposition(text: str) -> str:
    normalized = _strip_yes_no_prefix(text)
    patterns = (
        r"\bon top of\s+(.+)$",
        r"\bin front of\s+(.+)$",
        r"\bnext to\s+(.+)$",
        r"\bon\s+(.+)$",
        r"\bin\s+(.+)$",
        r"\bunder\s+(.+)$",
        r"\bbeside\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return _canonical_phrase(match.group(1))
    return _canonical_phrase(normalized)


def _detect_question_type(question: str) -> str:
    q = _normalize_question(question)
    if "color" in q:
        return "color"
    if "material" in q or "made of" in q or "constructed using" in q:
        return "material"
    if q.startswith("how many ") or "total number" in q or "count the number" in q:
        return "count"
    if any(
        phrase in q
        for phrase in (
            "left or right",
            "left side",
            "right side",
            "which side",
            "on which side",
            "top or the bottom",
            "top or bottom",
            "in front of",
            "behind",
            "above",
            "below",
        )
    ):
        return "spatial"
    if q.startswith("who "):
        return "who"
    if any(token in q for token in ("face expression", "facial expression", "state", "emotion", "activity", "pose")):
        return "attribute"
    if any(token in q for token in ("pattern", "shape", "texture", "size")):
        return "attribute"
    if q.startswith("what type of") or q.startswith("what kind of") or "what type " in q or "what kind " in q:
        return "predicate"
    if q.startswith(("is ", "are ", "does ", "do ", "can ")):
        return "yes_no"
    if q.startswith(_QUESTION_PATTERNS_SUBJECT) or q.startswith("what object") or q.startswith("what are some"):
        return "subject"
    if any(phrase in q for phrase in _QUESTION_PATTERNS_OBJECT):
        return "object"
    return "predicate"


def _extract_canonical_label(question: str, text: str) -> str:
    qtype = _detect_question_type(question)
    if qtype == "yes_no":
        return _extract_yes_no_label(text) or ""
    if qtype == "color":
        predicate = _extract_predicate_phrase(text)
        return _extract_vocab_label(predicate, _COLORS, _COLOR_ALIASES) or _extract_vocab_label(
            text, _COLORS, _COLOR_ALIASES
        ) or ""
    if qtype == "material":
        predicate = _extract_predicate_phrase(text)
        return _extract_vocab_label(predicate, _MATERIALS, _MATERIAL_ALIASES) or _extract_vocab_label(
            text, _MATERIALS, _MATERIAL_ALIASES
        ) or ""
    if qtype == "count":
        return _extract_number_label(text) or ""
    if qtype == "spatial":
        predicate = _extract_predicate_phrase(text)
        return _extract_spatial_label(predicate) or _extract_spatial_label(text) or ""
    if qtype == "attribute":
        return _extract_attribute_label(text) or _extract_predicate_phrase(text)
    if qtype == "who":
        return _extract_subject_phrase(text)
    if qtype == "subject":
        return _extract_subject_phrase(text)
    if qtype == "object":
        return _extract_object_after_preposition(text)
    return _extract_predicate_phrase(text)


def format_reward(response: str) -> float:
    _, found = _extract_answer_content(response)
    return 1.0 if found else 0.0


def accuracy_reward(response: str, ground_truth: str, question: str = "") -> float:
    answer_text, _ = _extract_answer_content(response)
    pred_label = _extract_canonical_label(question, answer_text)
    gt_label = _extract_canonical_label(question, ground_truth)
    if not pred_label or not gt_label:
        return 0.0
    return 1.0 if pred_label == gt_label else 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        question = reward_input.get("question", "")
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"], question=question)
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )
    return scores
