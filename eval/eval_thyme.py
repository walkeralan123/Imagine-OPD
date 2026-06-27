#!/usr/bin/env python3
"""
eval/eval_thyme.py

Thyme dataset evaluation helpers.
Uses the original label field as the category.
"""

import json
import os
import re


def load_data(data_path, image_dir):
    """Load Thyme data. Uses raw label values (text / vision) as category."""
    with open(data_path) as f:
        dataset = json.load(f)
    items = []
    for i, d in enumerate(dataset):
        # Get primary image (_0.png)
        image_list = d.get("image", [])
        primary_img = None
        for img_path in image_list:
            if img_path.endswith("_0.png"):
                full_path = os.path.join(image_dir, os.path.basename(img_path))
                if os.path.exists(full_path):
                    primary_img = full_path
                    break
        if not primary_img and image_list:
            full_path = os.path.join(image_dir, os.path.basename(image_list[0]))
            if os.path.exists(full_path):
                primary_img = full_path

        # Clean question
        raw_q = d.get("question", "")
        idx = raw_q.find("### User Image Path")
        if idx > 0:
            raw_q = raw_q[:idx]
        raw_q = raw_q.replace("<image>\n", "").replace("<image>", "").strip()

        # Extract GT answer from response
        gt_answer = None
        resp = d.get("response", "")
        if resp:
            ans_match = re.search(r'<answer>\s*(.*?)\s*</answer>', resp, re.DOTALL)
            if ans_match:
                gt_answer = ans_match.group(1).strip()

        # Use raw label as category (text / vision / unknown)
        category = d.get("label") or "unknown"

        items.append({
            "question": raw_q,
            "image_path": primary_img,
            "gt_label": gt_answer,
            "question_id": str(d.get("data_idx", i)),
            "category": category,
            "raw": d,
        })
    return items


def extract_answer(response_text):
    """Extract answer from model response for Thyme dataset."""
    if response_text is None:
        return None
    # Try <answer> tag first
    match = re.search(r'<answer>\s*(.*?)\s*(?:</answer>|$)', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try \\boxed{}
    match = re.search(r'\\boxed\{(.*?)\}', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()


def check_correct(pred_answer, gt_label):
    """Check correctness for Thyme dataset (free-form or option-based)."""
    if not pred_answer or not gt_label:
        return False
    pred = str(pred_answer).strip()
    gt = str(gt_label).strip()

    # Strip \\boxed{} wrapper from both sides
    for s in [pred, gt]:
        m = re.search(r'\\boxed\{(.*?)\}', s, re.DOTALL)
        if m:
            if s is pred:
                pred = m.group(1).strip()
            else:
                gt = m.group(1).strip()

    # Try option letter match (A-D)
    pred_letter = re.search(r'([A-D])', pred)
    gt_letter = re.search(r'([A-D])', gt)
    if pred_letter and gt_letter and len(gt) <= 3:
        return pred_letter.group(1) == gt_letter.group(1)

    # Exact match (case-insensitive)
    return pred.lower() == gt.lower()
