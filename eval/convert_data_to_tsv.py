#!/usr/bin/env python3
"""
eval/convert_data_to_tsv.py

Convert project-local data (data/*/data.json) into VLMEvalKit-compatible TSV
files so that VLMEvalKit can read them without downloading anything.

Supports two modes:
  - MCQ:  parses (A)/(B)/... options into separate columns  (e.g. vstar, hr_bench)
  - VQA:  free-form question + answer                       (e.g. thyme, monet)

"""

import argparse
import importlib
import json
import os
import re
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Reuse the project's dataset module registry
DATASET_MODULES = {
    "vstar": "eval.eval_vstar",
    "thyme": "eval.eval_thyme",
    "monet": "eval.eval_monet",
    "hr_bench": "eval.eval_hr_bench",
    "ocrbench": "eval.eval_ocrbench",
    "blink": None,
    "mme_realworld_lite": None,
    "cv_bench": None,
    "countqa": None,
    "babyvision": None,
    "treebench": None,
}

# MCQ datasets whose question text contains (A)/(B)/... style options
MCQ_DATASETS = {"vstar", "hr_bench", "blink", "mme_realworld_lite", "cv_bench", "treebench"}

OPTION_RE = re.compile(r"^\(([A-Z])\)\s*(.+)$")
MME_OPTION_RE = re.compile(r"^\(([A-Z])\)\s*(.*)$", re.DOTALL)


def parse_mcq_options(text):
    """Split question text that contains (A)...(B)... into question + options dict."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    question_lines = []
    options = {}
    for line in lines:
        m = OPTION_RE.match(line)
        if m:
            options[m.group(1)] = m.group(2).strip()
        elif line.lower().startswith("answer with"):
            continue
        else:
            question_lines.append(line)
    question = "\n".join(question_lines).strip()
    return question, options


def parse_mme_options(answer_choices):
    """Parse MME-RealWorld-Lite's option list into MCQ columns."""
    options = {}
    for raw_choice in answer_choices or []:
        choice = str(raw_choice).strip()
        match = MME_OPTION_RE.match(choice)
        if not match:
            continue
        label = match.group(1).strip()
        content = match.group(2).strip()
        if not content:
            continue
        options[label] = content
    return options


def load_mme_realworld_lite_items(data_path, image_dir):
    """Load MME-RealWorld-Lite directly from the normalized data.json."""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    items = []
    for item in raw_items:
        image_name = str(item.get("Image", "")).strip()
        question_id = str(item.get("Question_id", "")).strip()
        question = str(item.get("Text", "")).strip()
        answer = str(item.get("Ground truth", "")).strip()
        if not image_name or not question_id or not question or not answer:
            continue

        image_path = os.path.join(image_dir, image_name)
        category_parts = [
            str(item.get("Task", "")).strip(),
            str(item.get("Subtask", "")).strip(),
            str(item.get("Category", "")).strip(),
        ]
        items.append(
            {
                "question_id": question_id,
                "question": question,
                "gt_label": answer,
                "image_path": image_path,
                "category": "/".join(part for part in category_parts if part),
                "options": parse_mme_options(item.get("Answer choices", [])),
            }
        )
    return items


def load_blink_items(data_path, image_dir):
    """Load merged BLINK local data.json into the common MCQ item format."""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    items = []
    for item in raw_items:
        question_id = str(item.get("question_id", "")).strip()
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if answer.startswith("(") and answer.endswith(")") and len(answer) == 3:
            answer = answer[1]

        image_names = item.get("image_path_list") or []
        image_paths = [
            os.path.join(image_dir, str(name).strip())
            for name in image_names
            if str(name).strip()
        ]
        choices = item.get("choices") or []
        options = {
            chr(ord("A") + idx): str(choice).strip()
            for idx, choice in enumerate(choices)
            if str(choice).strip()
        }

        if not question_id or not question or not answer or not image_paths:
            continue

        items.append(
            {
                "question_id": question_id,
                "question": question,
                "gt_label": answer,
                "image_path_list": image_paths,
                "category": str(item.get("sub_task", "")).strip() or "unknown",
                "split": str(item.get("split", "")).strip() or "unknown",
                "options": options,
            }
        )
    return items


def load_cv_bench_items(data_path, image_dir):
    """Load local CV-Bench data.json into the common MCQ item format."""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    items = []
    for item in raw_items:
        question_id = str(item.get("question_id", "")).strip()
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if answer.startswith("(") and answer.endswith(")") and len(answer) == 3:
            answer = answer[1]

        image_names = item.get("image_path_list") or []
        image_paths = [
            os.path.join(image_dir, str(name).strip())
            for name in image_names
            if str(name).strip()
        ]
        choices = item.get("choices") or []
        options = {
            chr(ord("A") + idx): str(choice).strip()
            for idx, choice in enumerate(choices)
            if str(choice).strip()
        }

        if not question_id or not question or not answer or not image_paths:
            continue

        items.append(
            {
                "question_id": question_id,
                "question": question,
                "gt_label": answer,
                "image_path_list": image_paths,
                "split": str(item.get("split", "")).strip() or "test",
                "category": str(item.get("task", "")).strip() or "unknown",
                "l2-category": str(item.get("source", "")).strip() or "unknown",
                "cv_type": str(item.get("type", "")).strip() or "",
                "cv_source": str(item.get("source", "")).strip() or "",
                "cv_task": str(item.get("task", "")).strip() or "",
                "options": options,
            }
        )
    return items


def resolve_local_image_path(image_dir, image_name):
    """Resolve image paths that may be relative to either image_dir or dataset root."""
    image_name = str(image_name or "").strip()
    if not image_name:
        return ""
    if os.path.isabs(image_name):
        return image_name

    direct_path = os.path.join(image_dir, image_name)
    if os.path.exists(direct_path):
        return direct_path

    dataset_root = os.path.dirname(image_dir.rstrip(os.sep))
    root_relative_path = os.path.join(dataset_root, image_name)
    if os.path.exists(root_relative_path):
        return root_relative_path

    return direct_path


def load_countqa_items(data_path, image_dir):
    """Load local CountQA data.json into the common VQA item format."""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    items = []
    for pos, item in enumerate(raw_items):
        question_id = str(item.get("question_id", item.get("index", pos))).strip()
        question = str(item.get("question") or item.get("problem") or "").replace("<image>", "").strip()
        answer = str(item.get("answer", "")).strip()
        image_path = resolve_local_image_path(image_dir, item.get("image"))
        if not question_id or not question or not answer or not image_path:
            continue

        categories = item.get("categories") or []
        if isinstance(categories, list):
            category = "/".join(str(x).strip() for x in categories if str(x).strip())
        else:
            category = str(categories).strip()

        items.append(
            {
                "question_id": question_id,
                "question": question,
                "gt_label": answer,
                "image_path": image_path,
                "split": str(item.get("split", "")).strip() or "test",
                "category": category or "unknown",
                "countqa_original_index": item.get("original_index", ""),
                "countqa_question_index": item.get("question_index", ""),
                "countqa_is_focused": item.get("is_focused", ""),
            }
        )
    return items


def load_babyvision_items(data_path, image_dir):
    """Load local BabyVision data.json into the common VQA item format."""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    items = []
    for pos, item in enumerate(raw_items):
        question_id = str(item.get("question_id") or item.get("taskId") or item.get("index", pos)).strip()
        question = str(item.get("question") or item.get("problem") or "").replace("<image>", "").strip()
        answer = str(item.get("answer", "")).strip()
        image_path = resolve_local_image_path(image_dir, item.get("image"))
        if not question_id or not question or not answer or not image_path:
            continue

        options = item.get("options") or []
        ans_type = str(item.get("ansType", "")).strip()
        answer_letter = ""
        choice_ans = item.get("choiceAns")
        if ans_type == "choice" and choice_ans is not None:
            try:
                choice_idx = int(choice_ans)
                if 0 <= choice_idx < 26:
                    answer_letter = chr(ord("A") + choice_idx)
            except (TypeError, ValueError):
                answer_letter = ""

        if ans_type == "choice" and options:
            option_lines = [
                f"({chr(ord('A') + idx)}) {str(option).strip()}"
                for idx, option in enumerate(options)
                if str(option).strip()
            ]
            if option_lines:
                question = question + "\n" + "\n".join(option_lines) + "\nAnswer with the option's letter directly."

        items.append(
            {
                "question_id": question_id,
                "question": question,
                "gt_label": answer,
                "image_path": image_path,
                "split": str(item.get("split", "")).strip() or "train",
                "category": str(item.get("type", "")).strip() or "unknown",
                "l2-category": str(item.get("subtype", "")).strip() or "unknown",
                "babyvision_task_id": item.get("taskId", ""),
                "babyvision_ans_type": ans_type,
                "babyvision_answer_letter": answer_letter,
                "babyvision_choice_ans": item.get("choiceAns", ""),
                "babyvision_options": json.dumps(options, ensure_ascii=False),
            }
        )
    return items


def load_treebench_items(data_path, image_dir):
    """Load local TreeBench data.json into the common MCQ item format."""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    items = []
    for pos, item in enumerate(raw_items):
        question_id = str(item.get("index", pos)).strip()
        question = str(item.get("question") or item.get("problem") or "").replace("<image>", "").strip()
        answer = str(item.get("answer") or item.get("label") or "").strip()
        image_path = resolve_local_image_path(image_dir, item.get("image"))

        options_raw = item.get("options") or {}
        options = {}
        if isinstance(options_raw, dict):
            options = {
                str(key).strip().upper(): str(value).strip()
                for key, value in options_raw.items()
                if str(key).strip() and str(value).strip()
            }
        elif isinstance(item.get("choices"), list):
            options = {
                chr(ord("A") + idx): str(choice).strip()
                for idx, choice in enumerate(item.get("choices") or [])
                if str(choice).strip()
            }

        if not question_id or not question or not answer or not image_path:
            continue

        items.append(
            {
                "question_id": question_id,
                "question": question,
                "gt_label": answer,
                "image_path": image_path,
                "split": str(item.get("split", "")).strip() or "test",
                "category": str(item.get("category", "")).strip() or "unknown",
                "l2-category": str(item.get("l2-category", "")).strip() or "unknown",
                "options": options,
                "target_instances": json.dumps(item.get("target_instances") or [], ensure_ascii=False),
            }
        )
    return items


def convert(
    dataset_name,
    data_path,
    image_dir,
    output_tsv,
    ocrbench_include_original=False,
    ocrbench_original_data_path=None,
):
    """Load project data via the eval module and write a VLMEvalKit TSV."""
    # Resolve paths
    if not os.path.isabs(data_path):
        data_path = os.path.join(PROJECT_ROOT, data_path)
    if not os.path.isabs(image_dir):
        image_dir = os.path.join(PROJECT_ROOT, image_dir)

    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)

    # Load data via project eval module
    if dataset_name == "mme_realworld_lite":
        items = load_mme_realworld_lite_items(data_path, image_dir)
    elif dataset_name == "blink":
        items = load_blink_items(data_path, image_dir)
    elif dataset_name == "cv_bench":
        items = load_cv_bench_items(data_path, image_dir)
    elif dataset_name == "countqa":
        items = load_countqa_items(data_path, image_dir)
    elif dataset_name == "babyvision":
        items = load_babyvision_items(data_path, image_dir)
    elif dataset_name == "treebench":
        items = load_treebench_items(data_path, image_dir)
    else:
        mod = importlib.import_module(DATASET_MODULES[dataset_name])
        load_kwargs = {}
        if dataset_name == "ocrbench":
            load_kwargs = {
                "include_original_image": ocrbench_include_original,
                "original_data_path": ocrbench_original_data_path,
            }
        items = mod.load_data(data_path, image_dir, **load_kwargs)

    is_mcq = dataset_name in MCQ_DATASETS
    rows = []
    for idx, item in enumerate(items):
        image_path_list = item.get("image_path_list")
        if isinstance(image_path_list, list) and image_path_list:
            image_path_list = [
                os.path.abspath(path) if path and not os.path.isabs(path) else path
                for path in image_path_list
                if path
            ]
            image_path = json.dumps(image_path_list, ensure_ascii=False)
        else:
            image_path = item.get("image_path") or ""
            if image_path and not os.path.isabs(image_path):
                image_path = os.path.abspath(image_path)

        row = {
            "index": str(item.get("question_id", idx)),
            "answer": str(item.get("gt_label", "")),
            "image_path": image_path,
            "category": item.get("category", ""),
        }
        if item.get("split"):
            row["split"] = item.get("split")
        if item.get("l2-category"):
            row["l2-category"] = item.get("l2-category")
        if item.get("target_instances"):
            row["target_instances"] = item.get("target_instances")
        if item.get("cv_type"):
            row["cv_type"] = item.get("cv_type")
        if item.get("cv_source"):
            row["cv_source"] = item.get("cv_source")
        if item.get("cv_task"):
            row["cv_task"] = item.get("cv_task")
        for key in (
            "countqa_original_index",
            "countqa_question_index",
            "countqa_is_focused",
            "babyvision_task_id",
            "babyvision_ans_type",
            "babyvision_answer_letter",
            "babyvision_choice_ans",
            "babyvision_options",
        ):
            if key in item:
                row[key] = item.get(key)

        if is_mcq:
            if dataset_name in {"mme_realworld_lite", "blink", "cv_bench", "treebench"}:
                question = item["question"]
                options = item.get("options", {})
            else:
                question, options = parse_mcq_options(item["question"])
            row["question"] = question
            option_keys = "ABCDEFGHIJK" if dataset_name == "treebench" else "ABCDEF"
            for key in option_keys:
                row[key] = options.get(key)
        else:
            row["question"] = item["question"]

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_tsv, sep="\t", index=False)
    print(f"[convert_data_to_tsv] Wrote {len(df)} rows -> {output_tsv}")
    return output_tsv


def parse_args():
    p = argparse.ArgumentParser(description="Convert local data to VLMEvalKit TSV")
    p.add_argument("--dataset", required=True, choices=list(DATASET_MODULES.keys()))
    p.add_argument("--data_path", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output_tsv", required=True)
    p.add_argument(
        "--ocrbench_include_original",
        action="store_true",
        help="For OCRBench transformed variants, also provide the matching original image.",
    )
    p.add_argument(
        "--ocrbench_original_data_path",
        default=None,
        help="Optional path to OCRBench original data.json used to resolve the matching original image.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(
        args.dataset,
        args.data_path,
        args.image_dir,
        args.output_tsv,
        ocrbench_include_original=args.ocrbench_include_original,
        ocrbench_original_data_path=args.ocrbench_original_data_path,
    )
