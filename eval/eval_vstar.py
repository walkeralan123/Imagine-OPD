#!/usr/bin/env python3
"""
eval/eval_vstar.py

V*Bench dataset evaluation helpers.
Uses the original category field as the category.
"""

import json
import os
import re


def load_data(data_path, image_dir):
    """Load V*Bench data. Uses raw category values (direct_attributes / relative_position)."""
    with open(data_path) as f:
        dataset = json.load(f)
    items = []
    for d in dataset:
        image_path = os.path.join(image_dir, d["image"])
        items.append({
            "question": d["text"],
            "image_path": image_path,
            "gt_label": d.get("label"),
            "question_id": d.get("question_id", ""),
            "category": d.get("category", "unknown"),
            "raw": d,
        })
    return items


def extract_answer(response_text):
    """Extract answer from model response for V*Bench (single letter A-D)."""
    if response_text is None:
        return None
    # Try <answer> tag first
    match = re.search(r'<answer>\s*(.*?)\s*(?:</answer>|$)', response_text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        # Try \\boxed{}
        match = re.search(r'\\boxed\{(.*?)\}', response_text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            text = response_text.strip()
    # Normalize to single letter
    return _normalize_option(text)


def check_correct(pred_answer, gt_label):
    """Check correctness for V*Bench multiple-choice (A/B/C/D)."""
    if not pred_answer or not gt_label:
        return False
    pred = _normalize_option(str(pred_answer))
    gt = _normalize_option(str(gt_label))
    if pred and gt and re.fullmatch(r'[A-D]', pred) and re.fullmatch(r'[A-D]', gt):
        return pred == gt
    return str(pred).strip().lower() == str(gt).strip().lower()


def _normalize_option(text):
    """Normalize free-form text to a single option letter (A-D) when possible."""
    if not text:
        return text
    text = text.strip()
    # Strip boxed wrapper if present
    m = re.search(r'\\boxed\{(.*?)\}', text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # Try various patterns
    patterns = [
        r'^\(?\s*([A-D])\s*\)?[\s\.:,;-]*$',           # "A", "(A)", "A."
        r'^\(?\s*([A-D])\s*\)?\s+[A-Za-z].*$',          # "(B) left"
        r'(?:option|answer|final answer)\s*[:\-]?\s*\(?\s*([A-D])\s*\)?',
        r'^\s*([A-D])\s*$',                              # bare letter
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # Fallback: if exactly one A-D letter word appears
    letters = re.findall(r'\b([A-D])\b', text.upper())
    if len(letters) == 1:
        return letters[0]
    return text
