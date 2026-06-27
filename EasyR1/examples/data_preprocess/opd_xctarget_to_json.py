"""
Convert local OPD x/c/target JSON files to the normalized JSON schema expected by
examples/data_preprocess/opd_json.py.

This keeps image references as local paths so the output JSON stays compact. The
downstream opd_json.py script can then turn the normalized JSON into train.parquet.

"""

import argparse
import glob
import json
import os
import re
from collections import Counter
from typing import Any


ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*([A-Da-d])\s*</answer>", re.IGNORECASE)
MCQ_ANSWER_PATTERN = re.compile(r"[A-Da-d]")


def resolve_input_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        expanded = os.path.expanduser(pattern)
        matches = sorted(glob.glob(expanded))
        if matches:
            paths.extend(matches)
        elif os.path.exists(expanded):
            paths.append(expanded)
        else:
            raise FileNotFoundError(f"No input file matched: {pattern}")

    unique_paths = list(dict.fromkeys(os.path.abspath(path) for path in paths))
    if not unique_paths:
        raise ValueError("No input files resolved from --inputs.")
    return unique_paths


def normalize_paths(paths: Any, base_dir: str) -> list[str]:
    if paths is None:
        return []
    if isinstance(paths, str):
        raw_paths = [paths]
    else:
        raw_paths = [path for path in paths if path]

    normalized = []
    for path in raw_paths:
        if path.startswith("data:"):
            normalized.append(path)
            continue

        expanded = os.path.expanduser(path)
        if not os.path.isabs(expanded):
            expanded = os.path.abspath(os.path.join(base_dir, expanded))
        normalized.append(expanded)
    return normalized


def build_user_content(question: str, image_count: int) -> str:
    question = (question or "").strip()
    if image_count <= 0:
        return question

    prefix = "<image>" * image_count
    if question.lstrip().startswith("<image>") and question.count("<image>") >= image_count:
        return question
    return f"{prefix}{question}"


def build_assistant_content(target: dict[str, Any]) -> str:
    final_response = (target.get("final_response") or "").strip()
    if ANSWER_TAG_PATTERN.search(final_response):
        return final_response

    answer = str(target.get("answer") or "").strip()
    if not answer:
        return ""
    return f"<answer>{answer}</answer>"


def validate_mcq_answer(answer: str) -> bool:
    return bool(MCQ_ANSWER_PATTERN.fullmatch(answer.strip()))


def convert_sample(
    item: dict[str, Any],
    source_path: str,
    allow_non_mcq_answers: bool,
    skip_missing_images: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    source_dir = os.path.dirname(source_path)
    x = item.get("x") or {}
    c = item.get("c") or {}
    target = item.get("target") or {}

    question = (x.get("question") or "").strip()
    answer = str(target.get("answer") or "").strip()
    assistant_content = build_assistant_content(target)

    student_images = normalize_paths(x.get("images"), source_dir)
    teacher_extra = normalize_paths(c.get("intermediate_images"), source_dir)
    teacher_images = student_images + teacher_extra

    if not question or not answer or not student_images or not assistant_content:
        return None, "missing_required_fields"

    if not allow_non_mcq_answers and not validate_mcq_answer(answer):
        return None, "non_mcq_answer"

    missing_paths = [
        path for path in teacher_images if not path.startswith("data:") and not os.path.exists(path)
    ]
    if missing_paths:
        if skip_missing_images:
            return None, "missing_image"
        raise FileNotFoundError(
            f"Sample {item.get('sample_id') or '<unknown>'} references missing image(s): {missing_paths}"
        )

    user_content = build_user_content(question, len(student_images))
    record = {
        "sample_id": item.get("sample_id"),
        "source_json": source_path,
        "source_index": item.get("source_index"),
        "op_type": item.get("op_type"),
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": student_images,
        "teacher_images": teacher_images,
        "target_answer": answer,
    }
    return record, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more OPD x/c/target JSON files or glob patterns.",
    )
    parser.add_argument("--save_path", required=True, help="Path to the normalized output JSON file.")
    parser.add_argument(
        "--allow_non_mcq_answers",
        action="store_true",
        help="Keep samples whose target.answer is not a single A-D choice.",
    )
    parser.add_argument(
        "--skip_missing_images",
        action="store_true",
        help="Skip samples with missing image paths instead of stopping with an error.",
    )
    parser.add_argument(
        "--dedupe_by_sample_id",
        action="store_true",
        help="Drop later duplicates that share the same sample_id across merged inputs.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on total converted samples.")
    args = parser.parse_args()

    input_paths = resolve_input_paths(args.inputs)
    counters = Counter()
    normalized_records = []
    seen_sample_ids: set[str] = set()

    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise TypeError(f"Expected a list in {input_path}, got {type(data).__name__}")

        counters["input_files"] += 1
        counters["input_samples"] += len(data)

        for item in data:
            sample_id = item.get("sample_id")
            if args.dedupe_by_sample_id and sample_id:
                if sample_id in seen_sample_ids:
                    counters["skipped_duplicate_sample_id"] += 1
                    continue

            record, skip_reason = convert_sample(
                item,
                source_path=input_path,
                allow_non_mcq_answers=args.allow_non_mcq_answers,
                skip_missing_images=args.skip_missing_images,
            )
            if skip_reason is not None:
                counters[f"skipped_{skip_reason}"] += 1
                continue

            if sample_id:
                seen_sample_ids.add(sample_id)
            normalized_records.append(record)

            if args.limit is not None and len(normalized_records) >= args.limit:
                break

        if args.limit is not None and len(normalized_records) >= args.limit:
            break

    save_path = os.path.abspath(os.path.expanduser(args.save_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(normalized_records, f, ensure_ascii=False, indent=2)

    counters["written_samples"] = len(normalized_records)
    print(f"Resolved {len(input_paths)} input file(s)")
    print(f"Wrote {len(normalized_records)} normalized sample(s) to {save_path}")
    for key in sorted(counters):
        print(f"{key}: {counters[key]}")


if __name__ == "__main__":
    main()
