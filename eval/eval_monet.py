#!/usr/bin/env python3
"""
eval/eval_monet.py

Monet dataset evaluation helpers.
Uses metadata.dataset_name as the category.
"""

import json
import os
import re


def load_data(data_path, image_dir):
    """Load Monet data. Uses metadata.dataset_name as category."""
    with open(data_path) as f:
        dataset = json.load(f)
    items = []
    for i, d in enumerate(dataset):
        # Extract user question and image from conversations
        question = ""
        image_path = None
        conversations = d.get("conversations", [])
        for conv in conversations:
            if conv.get("role") == "user":
                for part in conv.get("content", []):
                    if isinstance(part, dict):
                        if part.get("type") == "text" and part.get("text"):
                            question = part["text"]
                        elif part.get("type") == "image" and part.get("image"):
                            image_path = os.path.join(image_dir, os.path.basename(part["image"]))

        # Get primary image from images list if not found in conversations
        if not image_path:
            images = d.get("images", [])
            if images:
                # images list contains filenames like "2_0_CogCoM_images_13"
                img_name = images[0]
                # Try common extensions
                for ext in [".png", ".jpg", ".jpeg"]:
                    candidate = os.path.join(image_dir, img_name + ext)
                    if os.path.exists(candidate):
                        image_path = candidate
                        break
                if not image_path:
                    image_path = os.path.join(image_dir, img_name + ".png")

        # Extract GT answer from assistant conversation
        gt_answer = None
        for conv in conversations:
            if conv.get("role") == "assistant":
                texts = []
                for part in conv.get("content", []):
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        texts.append(part["text"])
                full_text = " ".join(texts)
                # Try \\boxed{}
                m = re.search(r'\\boxed\{(.*?)\}', full_text, re.DOTALL)
                if m:
                    gt_answer = m.group(1).strip()
                else:
                    # Use last text segment as answer
                    if texts:
                        gt_answer = texts[-1].strip()

        # Use metadata.dataset_name as category
        category = d.get("metadata", {}).get("dataset_name", "unknown")

        items.append({
            "question": question,
            "image_path": image_path,
            "gt_label": gt_answer,
            "question_id": str(d.get("id", i)),
            "category": category,
            "raw": d,
        })
    return items


def extract_answer(response_text):
    """Extract answer from model response for Monet dataset."""
    if response_text is None:
        return None
    # Try <answer> tag
    match = re.search(r'<answer>\s*(.*?)\s*(?:</answer>|$)', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try \\boxed{}
    match = re.search(r'\\boxed\{(.*?)\}', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()


def check_correct(pred_answer, gt_label):
    """Check correctness for Monet dataset (free-form answers)."""
    if not pred_answer or not gt_label:
        return False
    pred = str(pred_answer).strip()
    gt = str(gt_label).strip()

    # Strip \\boxed{} wrapper
    m = re.search(r'\\boxed\{(.*?)\}', pred, re.DOTALL)
    if m:
        pred = m.group(1).strip()
    m = re.search(r'\\boxed\{(.*?)\}', gt, re.DOTALL)
    if m:
        gt = m.group(1).strip()

    # Exact match (case-insensitive)
    if pred.lower() == gt.lower():
        return True

    # Numeric match: try to compare as numbers
    try:
        pred_num = float(pred.replace(",", "").replace("%", ""))
        gt_num = float(gt.replace(",", "").replace("%", ""))
        if abs(pred_num - gt_num) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    # Containment: if gt is short, check if pred contains it
    if len(gt) <= 30 and gt.lower() in pred.lower():
        return True

    return False
