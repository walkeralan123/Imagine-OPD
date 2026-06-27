#!/usr/bin/env python3
"""
eval/eval_hr_bench.py

HR-Bench dataset evaluation helpers.
Uses the original category field as the category.
"""

import json
import os
import re


def _build_question_with_options(item):
    question = str(item.get("question", "")).strip()
    raw = item.get("raw", {}) if isinstance(item.get("raw"), dict) else {}

    options = []
    for key in ["A", "B", "C", "D"]:
        value = raw.get(key)
        if value is not None and str(value).strip():
            options.append(f"({key}) {str(value).strip()}")

    if not options:
        return question

    return (
        f"{question}\n"
        + "\n".join(options)
        + "\nAnswer with the option's letter from the given choices directly."
    )


def load_data(data_path, image_dir):
    """Load HR-Bench data from normalized local JSON."""
    with open(data_path) as f:
        dataset = json.load(f)

    items = []
    for i, d in enumerate(dataset):
        image_name = d.get("image", "")
        image_path = None
        if image_name:
            if os.path.isabs(image_name):
                image_path = image_name
            elif image_name.startswith("images/") or image_name.startswith("images\\"):
                image_path = os.path.join(os.path.dirname(data_path), image_name)
            else:
                image_path = os.path.join(image_dir, image_name)
        items.append({
            "question": _build_question_with_options(d),
            "image_path": image_path,
            "gt_label": d.get("label"),
            "question_id": str(d.get("question_id", i)),
            "split": d.get("split", "unknown"),
            "category": d.get("category", "unknown"),
            "raw": d,
        })
    return items


def extract_answer(response_text):
    """Extract answer from model response for HR-Bench."""
    if response_text is None:
        return None

    match = re.search(r'<answer>\s*(.*?)\s*(?:</answer>|$)', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r'\\boxed\{(.*?)\}', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return response_text.strip()


def check_correct(pred_answer, gt_label):
    """Check correctness for HR-Bench free-form or option-based answers."""
    if not pred_answer or not gt_label:
        return False

    pred = _normalize_text(pred_answer)
    gt = _normalize_text(gt_label)

    pred_option = _extract_option(pred)
    gt_option = _extract_option(gt)
    if pred_option and gt_option:
        return pred_option == gt_option

    if pred == gt:
        return True

    try:
        pred_num = float(pred.replace(",", "").replace("%", ""))
        gt_num = float(gt.replace(",", "").replace("%", ""))
        return abs(pred_num - gt_num) < 1e-6
    except (ValueError, TypeError):
        pass

    return False


def _normalize_text(text):
    text = str(text).strip()
    m = re.search(r'\\boxed\{(.*?)\}', text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return text.lower()


def _extract_option(text):
    m = re.search(r'\b([A-D])\b', str(text).upper())
    if m:
        return m.group(1)
    return None
