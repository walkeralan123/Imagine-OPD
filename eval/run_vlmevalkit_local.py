#!/usr/bin/env python3
"""
eval/run_vlmevalkit_local.py

Unified entry point for running VLMEvalKit evaluations using the project's
local resources:
  - api_config_files/*.json   → model endpoint
  - prompt_template/*.json    → prompt template
  - data/*/data.json          → dataset

This script:
  1. Converts local data to VLMEvalKit TSV (avoids any download)
  2. Generates a VLMEvalKit config JSON on the fly
  3. Registers the custom ImagineModel into VLMEvalKit's namespace
  4. Calls VLMEvalKit's main() function
"""

import argparse
import glob
import json
import os
import re
import sys
import tempfile
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_LMUDATA_DIR = os.path.join(PROJECT_ROOT, ".cache", "LMUData")

VLMEVAL_ROOT = os.environ.get(
    "VLMEVAL_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "VLMEvalKit"),
)
os.environ.setdefault("VLMEVAL_ROOT", VLMEVAL_ROOT)
if VLMEVAL_ROOT not in sys.path:
    sys.path.insert(0, VLMEVAL_ROOT)

# Dataset name → VLMEvalKit benchmark name (used for TSV filename)
VLMEVAL_DATASET_MAP = {
    "vstar": ("VStarBench", "LocalImageMCQDataset"),
    "hr_bench": ("HRBench", "LocalImageMCQDataset"),
    "thyme": ("ThymeBench", "LocalImageVQADataset"),
    "monet": ("MonetBench", "LocalImageVQADataset"),
    "ocrbench": ("OCRBench", "LocalOCRBenchDataset"),
    "blink": ("BLINK", "LocalImageMCQDataset"),
    "mme_realworld_lite": ("MMERealWorldLite", "LocalImageMCQDataset"),
    "cv_bench": ("CVBench", "LocalImageMCQDataset"),
    "countqa": ("CountQABench", "LocalCountQADataset"),
    "babyvision": ("BabyVisionBench", "LocalBabyVisionDataset"),
    "treebench": ("TreeBench", "LocalTreeBenchDataset"),
}

# Local shorthand -> official VLMEvalKit supported_VLM name.
# This can be enabled explicitly when we want native VLMEvalKit model loading.
OFFICIAL_MODEL_ALIASES = {
    "qwen": "Qwen2.5-VL-7B-Instruct",
}

OFFICIAL_MODEL_PATHS = {
    "qwen": os.environ.get("QWEN25_VL_7B_MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct"),
}

ANSWER_KEY_CANDIDATES = {
    "mme_realworld_lite": ["Ground truth", "ground_truth", "answer", "label"],
    "vstar": ["label", "answer", "gt_label"],
    "hr_bench": ["label", "answer", "gt_label"],
    "cv_bench": ["answer", "label", "gt_label"],
    "countqa": ["answer", "label", "gt_label"],
    "babyvision": ["answer", "blankAns", "label", "gt_label"],
    "treebench": ["answer", "label", "gt_label"],
}

INDEX_KEY_CANDIDATES = {
    "mme_realworld_lite": ["Question_id", "question_id", "index", "id"],
    "vstar": ["question_id", "index", "id"],
    "hr_bench": ["question_id", "index", "id"],
    "cv_bench": ["question_id", "index", "id"],
    "countqa": ["question_id", "index", "id"],
    "babyvision": ["question_id", "taskId", "index", "id"],
    "treebench": ["index", "question_id", "id"],
}

COMMON_ANSWER_KEYS = [
    "answer",
    "label",
    "gt_label",
    "Ground truth",
    "ground_truth",
    "ground truth",
    "target_answer",
    "correct_answer",
    "correct",
]

COMMON_INDEX_KEYS = ["question_id", "Question_id", "index", "id"]
ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(.*?)(?:</answer>|$)", re.IGNORECASE | re.DOTALL)
FIRST_LETTER_PATTERN = re.compile(r"[A-Za-z]")

SUBGROUP_REPORT_CONFIGS = {
    "vstar": {
        "name": "VStar",
        "field_candidates": ["category"],
        "groups": [
            ("direct_attributes", "direct_attributes"),
            ("relative_position", "relative_position"),
        ],
    },
    "hr_bench": {
        "name": "HRBench",
        "field_candidates": ["category"],
        "groups": [
            ("single", "single"),
            ("cross", "cross"),
        ],
    },
    "mme_realworld_lite": {
        "name": "MME-RealWorld-Lite",
        "field_candidates": ["Task"],
        "groups": [
            ("reasoning", "Reasoning"),
            ("perception", "Perception"),
        ],
    },
    "cv_bench": {
        "name": "CVBench",
        "field_candidates": ["type", "cv_type"],
        "groups": [
            ("2D", "2D"),
            ("3D", "3D"),
        ],
    },
    "treebench": {
        "name": "TreeBench",
        "field_candidates": ["category"],
        "groups": [
            ("reasoning", "Reasoning"),
            ("perception", "Perception"),
        ],
    },
}


def resolve(path):
    """Resolve a path relative to PROJECT_ROOT if not absolute."""
    if path and not os.path.isabs(path):
        return os.path.join(PROJECT_ROOT, path)
    return path


def _extract_first_nonempty(container: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        if key not in container:
            continue
        value = container.get(key)
        if isinstance(value, dict):
            continue
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _extract_answer_from_raw_record(record: dict[str, Any], dataset: str) -> str | None:
    if not isinstance(record, dict):
        return None

    containers = [record]
    raw = record.get("raw")
    if isinstance(raw, dict):
        containers.append(raw)
    target = record.get("target")
    if isinstance(target, dict):
        containers.append(target)

    candidate_keys = ANSWER_KEY_CANDIDATES.get(dataset, []) + COMMON_ANSWER_KEYS
    for container in containers:
        answer = _extract_first_nonempty(container, candidate_keys)
        if answer is not None:
            return answer
    return None


def _raw_record_containers(record: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(record, dict):
        return []

    containers = [record]
    raw = record.get("raw")
    if isinstance(raw, dict):
        containers.append(raw)
    target = record.get("target")
    if isinstance(target, dict):
        containers.append(target)
    return containers


def _extract_metadata_value_from_raw_record(
    record: dict[str, Any], candidate_keys: list[str]
) -> str | None:
    for container in _raw_record_containers(record):
        value = _extract_first_nonempty(container, candidate_keys)
        if value is not None:
            return value
    return None


def _extract_index_from_raw_record(record: dict[str, Any], dataset: str, fallback: int) -> str:
    if not isinstance(record, dict):
        return str(fallback)

    candidate_keys = INDEX_KEY_CANDIDATES.get(dataset, []) + COMMON_INDEX_KEYS
    index_value = _extract_first_nonempty(record, candidate_keys)
    if index_value is not None:
        return index_value
    return str(fallback)


def _extract_subgroup_from_raw_record(record: dict[str, Any], dataset: str) -> str | None:
    config = SUBGROUP_REPORT_CONFIGS.get(dataset)
    if config is None:
        return None
    value = _extract_metadata_value_from_raw_record(record, config["field_candidates"])
    if dataset == "treebench" and value:
        return value.split("/", 1)[0].strip()
    return value


def _build_record_metadata_lookup(
    dataset: str, data_path: str
) -> tuple[dict[str, dict[str, str | None]], list[dict[str, str | None]]]:
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise TypeError(f"Expected a list in {data_path}, got {type(raw_data).__name__}")

    metadata_by_index: dict[str, dict[str, str | None]] = {}
    metadata_by_position: list[dict[str, str | None]] = []
    for pos, record in enumerate(raw_data):
        metadata = {
            "answer": _extract_answer_from_raw_record(record, dataset),
            "subgroup": _extract_subgroup_from_raw_record(record, dataset),
        }
        metadata_by_position.append(metadata)
        record_index = _extract_index_from_raw_record(record, dataset, fallback=pos)
        metadata_by_index[str(record_index)] = metadata
    return metadata_by_index, metadata_by_position


def _build_answer_lookup(dataset: str, data_path: str) -> tuple[dict[str, str], list[str | None]]:
    metadata_by_index, metadata_by_position = _build_record_metadata_lookup(dataset, data_path)
    answer_by_index: dict[str, str] = {}
    answer_by_position: list[str | None] = [meta.get("answer") for meta in metadata_by_position]
    for record_index, metadata in metadata_by_index.items():
        answer = metadata.get("answer")
        if answer is not None:
            answer_by_index[str(record_index)] = answer
    return answer_by_index, answer_by_position


def _infer_result_file_path(work_dir: str, display_name: str, vlmeval_name: str) -> str:
    from vlmeval.smp.file import get_pred_file_path

    return get_pred_file_path(
        work_dir=work_dir,
        model_name=display_name,
        dataset_name=vlmeval_name,
        use_env_format=True,
    )


def _resolve_existing_artifact_path(path: str) -> str | None:
    if os.path.exists(path):
        return path

    basename = os.path.basename(path)
    search_root = os.path.dirname(path) or "."
    search_pattern = os.path.join(search_root, "**", basename)
    matches = [match for match in glob.glob(search_pattern, recursive=True) if os.path.isfile(match)]
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def _ensure_infer_result_answers(args, work_dir: str, display_name: str, vlmeval_name: str) -> None:
    if args.mode == "eval":
        return

    result_file = _resolve_existing_artifact_path(
        _infer_result_file_path(work_dir, display_name, vlmeval_name)
    )
    if result_file is None:
        print("[run_vlmevalkit_local] Result file not found, skip answer backfill.")
        return
    if not result_file.endswith(".json"):
        print(
            f"[run_vlmevalkit_local] Skip answer backfill because result file is not JSON: {result_file}"
        )
        return
    if not os.path.exists(result_file):
        print(f"[run_vlmevalkit_local] Result JSON not found, skip answer backfill: {result_file}")
        return

    with open(result_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        print(
            f"[run_vlmevalkit_local] Skip answer backfill because result JSON is not a list: {result_file}"
        )
        return

    answer_by_index, answer_by_position = _build_answer_lookup(args.dataset, resolve(args.data_path))

    filled = 0
    missing = 0
    for pos, row in enumerate(records):
        if not isinstance(row, dict):
            continue
        current_answer = row.get("answer")
        if current_answer is not None and str(current_answer).strip():
            continue

        answer = None
        row_index = row.get("index")
        if row_index is not None:
            answer = answer_by_index.get(str(row_index))
        if answer is None and pos < len(answer_by_position):
            answer = answer_by_position[pos]

        if answer is None:
            missing += 1
            continue

        row["answer"] = answer
        filled += 1

    if filled == 0:
        print(
            f"[run_vlmevalkit_local] Answer backfill finished: no missing answers needed updates "
            f"(unresolved={missing})."
        )
        return

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

    print(
        f"[run_vlmevalkit_local] Added answer field to {filled} infer JSON records "
        f"(unresolved={missing}) -> {result_file}"
    )


def _iter_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_strings(value)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_strings(value)


def _extract_first_letter(text: str | None) -> str | None:
    if text is None:
        return None
    match = FIRST_LETTER_PATTERN.search(str(text))
    return match.group(0).upper() if match else None


def _extract_pred_letter_from_extra_records(extra_records: Any) -> str | None:
    last_answer_block = None
    for text in _iter_strings(extra_records):
        for match in ANSWER_TAG_PATTERN.finditer(text):
            last_answer_block = match.group(1)

    if last_answer_block is None:
        return None
    return _extract_first_letter(last_answer_block)


def _normalize_group_value(value: Any) -> str:
    return str(value).strip().lower()


def _init_accuracy_stats() -> dict[str, int]:
    return {
        "total": 0,
        "comparable": 0,
        "correct": 0,
        "missing_label": 0,
        "missing_prediction": 0,
    }


def _update_accuracy_stats(
    stats: dict[str, int], label: str | None, pred: str | None
) -> bool | None:
    stats["total"] += 1
    if label is None:
        stats["missing_label"] += 1
        return None
    if pred is None:
        stats["missing_prediction"] += 1
        return None

    is_correct = pred == label
    stats["comparable"] += 1
    stats["correct"] += int(is_correct)
    return is_correct


def _finalize_accuracy_stats(stats: dict[str, int]) -> dict[str, int | float]:
    finalized = dict(stats)
    comparable = finalized["comparable"]
    finalized["accuracy"] = (finalized["correct"] / comparable) if comparable else 0.0
    return finalized


def _format_accuracy_stats(stats: dict[str, int | float]) -> str:
    return (
        f"{stats['accuracy']:.4f} ({stats['correct']}/{stats['comparable']}), "
        f"total={stats['total']}, missing_label={stats['missing_label']}, "
        f"missing_prediction={stats['missing_prediction']}"
    )


def _resolve_row_metadata(
    row: dict[str, Any],
    pos: int,
    metadata_by_index: dict[str, dict[str, str | None]],
    metadata_by_position: list[dict[str, str | None]],
) -> dict[str, str | None]:
    row_index = row.get("index")
    if row_index is not None:
        metadata = metadata_by_index.get(str(row_index))
        if metadata is not None:
            return metadata
    if pos < len(metadata_by_position):
        return metadata_by_position[pos]
    return {}


def _report_infer_accuracy_from_extra_records(
    args, work_dir: str, display_name: str, vlmeval_name: str
) -> None:
    if args.mode != "infer":
        return

    result_file = _resolve_existing_artifact_path(
        _infer_result_file_path(work_dir, display_name, vlmeval_name)
    )
    if result_file is None:
        print("[run_vlmevalkit_local] Skip infer accuracy report because result file was not found.")
        return

    extra_records_file = _resolve_existing_artifact_path(
        os.path.splitext(result_file)[0] + "_extra_records.json"
    )
    if extra_records_file is None:
        print(
            f"[run_vlmevalkit_local] Skip infer accuracy report because extra records file "
            f"does not exist for result file: {result_file}"
        )
        return

    with open(extra_records_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        print(
            f"[run_vlmevalkit_local] Skip infer accuracy report because extra records JSON "
            f"is not a list: {extra_records_file}"
        )
        return

    metadata_by_index, metadata_by_position = _build_record_metadata_lookup(
        args.dataset, resolve(args.data_path)
    )
    subgroup_config = SUBGROUP_REPORT_CONFIGS.get(args.dataset)
    subgroup_stats: dict[str, dict[str, int]] = {}
    subgroup_key_to_name: dict[str, str] = {}
    if subgroup_config is not None:
        for display_name_, raw_value in subgroup_config["groups"]:
            normalized = _normalize_group_value(raw_value)
            subgroup_stats[display_name_] = _init_accuracy_stats()
            subgroup_key_to_name[normalized] = display_name_

    overall_stats = _init_accuracy_stats()
    unknown_subgroup_stats = _init_accuracy_stats()

    for pos, row in enumerate(records):
        if not isinstance(row, dict):
            continue

        label = _extract_first_letter(row.get("label"))
        metadata = _resolve_row_metadata(row, pos, metadata_by_index, metadata_by_position)
        if label is None:
            label = _extract_first_letter(metadata.get("answer"))

        pred = _extract_pred_letter_from_extra_records(row.get("extra_records"))
        update_result = _update_accuracy_stats(overall_stats, label, pred)

        if subgroup_config is not None:
            subgroup_value = metadata.get("subgroup")
            subgroup_name = subgroup_key_to_name.get(_normalize_group_value(subgroup_value))
            if subgroup_name is None:
                _update_accuracy_stats(unknown_subgroup_stats, label, pred)
            else:
                stats = subgroup_stats[subgroup_name]
                stats["total"] += 1
                if update_result is None:
                    if label is None:
                        stats["missing_label"] += 1
                    else:
                        stats["missing_prediction"] += 1
                else:
                    stats["comparable"] += 1
                    stats["correct"] += int(update_result)

    overall_summary = _finalize_accuracy_stats(overall_stats)
    subgroup_summary = {
        name: _finalize_accuracy_stats(stats)
        for name, stats in subgroup_stats.items()
    }
    unknown_subgroup_summary = _finalize_accuracy_stats(unknown_subgroup_stats)
    summary = {
        "dataset": args.dataset,
        "overall": overall_summary,
        "by_subgroup": subgroup_summary,
        "unknown_subgroup": unknown_subgroup_summary,
        "file": extra_records_file,
    }
    summary_path = os.path.splitext(extra_records_file)[0] + "_infer_accuracy_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        "[run_vlmevalkit_local] Infer accuracy from extra_records.json: "
        f"overall={_format_accuracy_stats(overall_summary)}, "
        f"summary_file={summary_path}, file={extra_records_file}"
    )
    if subgroup_config is not None:
        pieces = [
            f"{name}={_format_accuracy_stats(stats)}"
            for name, stats in subgroup_summary.items()
        ]
        if unknown_subgroup_summary["total"]:
            pieces.append(f"unknown={_format_accuracy_stats(unknown_subgroup_summary)}")
        print(
            f"[run_vlmevalkit_local] {subgroup_config['name']} subgroup accuracy: "
            + "; ".join(pieces)
        )


def _build_cv_bench_metadata(args) -> dict[str, dict[str, Any]]:
    data_path = resolve(args.data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    metadata_by_index: dict[str, dict[str, Any]] = {}
    for pos, record in enumerate(raw_data):
        record_index = _extract_index_from_raw_record(record, args.dataset, fallback=pos)
        metadata_by_index[str(record_index)] = {
            "answer": _extract_answer_from_raw_record(record, args.dataset),
            "type": str(record.get("type", "")).strip(),
            "task": str(record.get("task", "")).strip(),
            "source": str(record.get("source", "")).strip(),
            "source_dataset": str(record.get("source_dataset", "")).strip(),
        }
    return metadata_by_index


def _compute_cv_bench_metrics_from_extra_records(records: list[dict[str, Any]], metadata_by_index: dict[str, dict[str, Any]]):
    total = 0
    comparable = 0
    correct = 0
    missing_label = 0
    missing_prediction = 0
    by_source: dict[str, dict[str, int]] = {}
    by_type: dict[str, dict[str, int]] = {}
    by_task: dict[str, dict[str, int]] = {}

    for row in records:
        if not isinstance(row, dict):
            continue

        row_index = str(row.get("index"))
        meta = metadata_by_index.get(row_index, {})
        label = _extract_first_letter(row.get("label"))
        if label is None:
            label = _extract_first_letter(meta.get("answer"))

        pred = _extract_pred_letter_from_extra_records(row.get("extra_records"))
        total += 1
        if label is None:
            missing_label += 1
            continue
        if pred is None:
            missing_prediction += 1
            continue

        comparable += 1
        is_correct = pred == label
        correct += int(is_correct)

        source = meta.get("source") or "unknown"
        cv_type = meta.get("type") or "unknown"
        task = meta.get("task") or "unknown"

        for bucket, key in ((by_source, source), (by_type, cv_type), (by_task, task)):
            stats = bucket.setdefault(key, {"correct": 0, "total": 0})
            stats["total"] += 1
            stats["correct"] += int(is_correct)

    def _finalize(stats: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
        finalized = {}
        for key, value in sorted(stats.items()):
            acc = (value["correct"] / value["total"]) if value["total"] else None
            finalized[key] = {
                "correct": value["correct"],
                "total": value["total"],
                "accuracy": acc,
            }
        return finalized

    by_source_final = _finalize(by_source)
    by_type_final = _finalize(by_type)
    by_task_final = _finalize(by_task)

    ade_acc = by_source_final.get("ADE20K", {}).get("accuracy")
    coco_acc = by_source_final.get("COCO", {}).get("accuracy")
    omni_acc = by_source_final.get("Omni3D", {}).get("accuracy")
    two_d_avg = None
    cv_bench_score = None
    if ade_acc is not None and coco_acc is not None:
        two_d_avg = (ade_acc + coco_acc) / 2
    if two_d_avg is not None and omni_acc is not None:
        cv_bench_score = (two_d_avg + omni_acc) / 2

    return {
        "total": total,
        "comparable": comparable,
        "correct": correct,
        "accuracy": (correct / comparable) if comparable else 0.0,
        "missing_label": missing_label,
        "missing_prediction": missing_prediction,
        "by_source": by_source_final,
        "by_type": by_type_final,
        "by_task": by_task_final,
        "official_cv_bench": {
            "ade20k_accuracy": ade_acc,
            "coco_accuracy": coco_acc,
            "omni3d_accuracy": omni_acc,
            "two_d_average": two_d_avg,
            "cv_bench_accuracy": cv_bench_score,
        },
    }


def _report_cv_bench_summary_from_extra_records(
    args, work_dir: str, display_name: str, vlmeval_name: str
) -> None:
    if args.dataset != "cv_bench":
        return

    result_file = _resolve_existing_artifact_path(
        _infer_result_file_path(work_dir, display_name, vlmeval_name)
    )
    if result_file is None:
        print("[run_vlmevalkit_local] Skip CV-Bench summary because result file was not found.")
        return

    extra_records_file = _resolve_existing_artifact_path(
        os.path.splitext(result_file)[0] + "_extra_records.json"
    )
    if extra_records_file is None:
        print("[run_vlmevalkit_local] Skip CV-Bench summary because extra_records.json was not found.")
        return

    with open(extra_records_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        print(
            f"[run_vlmevalkit_local] Skip CV-Bench summary because extra records JSON "
            f"is not a list: {extra_records_file}"
        )
        return

    metadata_by_index = _build_cv_bench_metadata(args)
    summary = _compute_cv_bench_metrics_from_extra_records(records, metadata_by_index)

    summary_path = os.path.splitext(extra_records_file)[0] + "_cv_bench_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    official = summary["official_cv_bench"]
    cv_bench_accuracy = official.get("cv_bench_accuracy")
    cv_bench_text = f"{cv_bench_accuracy:.4f}" if cv_bench_accuracy is not None else "n/a"
    print(
        "[run_vlmevalkit_local] CV-Bench summary from extra_records.json: "
        f"overall={summary['accuracy']:.4f} ({summary['correct']}/{summary['comparable']}), "
        f"CV-Bench={cv_bench_text}, summary_file={summary_path}"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Run VLMEvalKit with local data, model, and prompt"
    )
    # Data
    p.add_argument("--dataset", required=True, choices=list(VLMEVAL_DATASET_MAP.keys()),
                    help="Dataset name (determines loader and VLMEvalKit benchmark class)")
    p.add_argument("--data_path", required=True, help="Path to data JSON (e.g. data/vstar/data.json)")
    p.add_argument("--image_dir", required=True, help="Path to image directory")
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
    # Model
    p.add_argument("--api_config", required=True, help="Path to API config JSON")
    p.add_argument("--model_name", default="qwen", help="Model name in vLLM (default: qwen)")
    p.add_argument("--client_type", default="openai", choices=["openai", "vllm", "anthropic"])
    p.add_argument("--display_name", default=None,
                    help="Display name for the model in results (auto-generated if not set)")
    p.add_argument(
        "--use_official_model",
        action="store_true",
        help="Use VLMEvalKit native supported_VLM loading instead of the local API bridge",
    )
    # Prompt
    p.add_argument("--prompt_template", required=True, help="Path to prompt template JSON")
    p.add_argument("--prompt", required=True, help="Prompt key in template JSON (e.g. no_tool, adaptive_vstar)")
    # Inference
    p.add_argument("--exe_code", action="store_true", help="Enable sandbox code execution")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--max_rounds", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    # VLMEvalKit
    p.add_argument("--work_dir", default=None, help="VLMEvalKit work dir (auto-generated if not set)")
    p.add_argument("--mode", default="all", choices=["all", "infer", "eval"])
    p.add_argument("--reuse", action="store_true", help="Reuse previous predictions")
    p.add_argument("--api_nproc", type=int, default=8, help="Number of API processes")
    p.add_argument("--judge", default=None,
                   help="Judge model name passed through to VLMEvalKit (e.g. gpt-4o-mini)")
    p.add_argument("--judge_args", default=None,
                   help="Judge arguments in JSON string format passed through to VLMEvalKit")
    p.add_argument(
        "--eval_method",
        default="rule",
        choices=["rule", "llm_judge"],
        help=(
            "Evaluation method for CountQA/BabyVision free-form answer accuracy. "
            "'rule' uses local matching; 'llm_judge' uses the configured judge model."
        ),
    )
    p.add_argument(
        "--use_llm_judge",
        action="store_true",
        help="Deprecated alias for --eval_method llm_judge.",
    )
    p.add_argument(
        "--llm_judge_api_config",
        default="api_config_files/api_config_openai.json",
        help="OpenAI-compatible API config JSON for --eval_method llm_judge.",
    )
    p.add_argument(
        "--llm_judge_model",
        default="claude-haiku-4-5-20251001",
        help="Judge model name for --eval_method llm_judge.",
    )
    args = p.parse_args()
    if args.use_llm_judge:
        args.eval_method = "llm_judge"
        print(
            "[run_vlmevalkit_local] Warning: --use_llm_judge is deprecated; "
            "use --eval_method llm_judge instead."
        )
    if args.mode == "infer" and args.eval_method == "llm_judge":
        print(
            "[run_vlmevalkit_local] Warning: --eval_method llm_judge has no effect "
            "when --mode infer is used because no evaluation stage runs."
        )
    return args


def step1_convert_data(
    dataset,
    data_path,
    image_dir,
    ocrbench_include_original=False,
    ocrbench_original_data_path=None,
):
    """Convert local data to VLMEvalKit TSV. Returns the TSV path."""
    from eval.convert_data_to_tsv import convert

    vlmeval_name, _ = VLMEVAL_DATASET_MAP[dataset]
    lmu_root = os.environ.get("LMUData", DEFAULT_LMUDATA_DIR)
    os.makedirs(lmu_root, exist_ok=True)
    output_tsv = os.path.join(lmu_root, f"{vlmeval_name}.tsv")

    convert_kwargs = {}
    if dataset == "ocrbench":
        convert_kwargs = {
            "ocrbench_include_original": ocrbench_include_original,
            "ocrbench_original_data_path": resolve(ocrbench_original_data_path),
        }

    convert(dataset, data_path, image_dir, output_tsv, **convert_kwargs)
    return output_tsv, vlmeval_name


def step2_generate_config(args, vlmeval_name, tsv_path):
    """Generate a VLMEvalKit config JSON and return its path."""
    _, dataset_class = VLMEVAL_DATASET_MAP[args.dataset]

    official_model_name = OFFICIAL_MODEL_ALIASES.get(args.model_name, args.model_name)
    use_official_model = False
    if args.use_official_model:
        from vlmeval.config import supported_VLM
        use_official_model = official_model_name in supported_VLM
    official_model_path = OFFICIAL_MODEL_PATHS.get(args.model_name)

    data_cfg = {
        "class": dataset_class,
        "dataset": vlmeval_name,
        "local_tsv": tsv_path,
    }
    if args.dataset in {"countqa", "babyvision"}:
        data_cfg.update(
            {
                "eval_method": args.eval_method,
                "llm_judge_api_config_path": resolve(args.llm_judge_api_config),
                "llm_judge_model": args.llm_judge_model,
            }
        )

    if use_official_model:
        display_name = args.display_name or official_model_name
    else:
        display_name = args.display_name or f"{args.model_name}_{args.prompt}"
    if args.exe_code and not use_official_model:
        display_name += "_sandbox"

    if use_official_model:
        print(
            f"[run_vlmevalkit_local] Using official VLMEvalKit model path: "
            f"{official_model_name}"
        )
        model_cfg = {}
        if official_model_path and os.path.exists(official_model_path):
            model_cfg["model_path"] = official_model_path
            print(
                f"[run_vlmevalkit_local] Using local model weights: "
                f"{official_model_path}"
            )
        print(
            "[run_vlmevalkit_local] Ignoring local API/prompt sandbox settings "
            "so inference matches the native VLMEvalKit model implementation"
        )
        config = {
            "model": {
                official_model_name: model_cfg
            },
            "data": {
                vlmeval_name: data_cfg
            },
        }
    else:
        config = {
            "model": {
                display_name: {
                    "class": "ImagineModel",
                    "api_config_path": resolve(args.api_config),
                    "prompt_template_path": resolve(args.prompt_template),
                    "prompt_key": args.prompt,
                    "model_name": args.model_name,
                    "client_type": args.client_type,
                    "exe_code": args.exe_code,
                    "max_tokens": args.max_tokens,
                    "max_rounds": args.max_rounds,
                    "temperature": args.temperature,
                }
            },
            "data": {
                vlmeval_name: data_cfg
            },
        }

    config_dir = os.path.join(PROJECT_ROOT, "eval", "vlmevalkit_configs")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"_auto_{display_name}_{vlmeval_name}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[run_vlmevalkit_local] Config written to {config_path}")
    return config_path, display_name


def step3_register_components():
    """Register local model and dataset adapters into VLMEvalKit namespaces."""
    import vlmeval.api
    import vlmeval.dataset
    from eval.vlmeval_model import ImagineModel
    from eval.vlmeval_local_dataset import (
        LocalBabyVisionDataset,
        LocalCountQADataset,
        LocalImageMCQDataset,
        LocalImageVQADataset,
        LocalOCRBenchDataset,
        LocalTreeBenchDataset,
    )

    vlmeval.api.ImagineModel = ImagineModel
    vlmeval.api.AdatwiModel = ImagineModel
    vlmeval.dataset.LocalBabyVisionDataset = LocalBabyVisionDataset
    vlmeval.dataset.LocalCountQADataset = LocalCountQADataset
    vlmeval.dataset.LocalImageMCQDataset = LocalImageMCQDataset
    vlmeval.dataset.LocalImageVQADataset = LocalImageVQADataset
    vlmeval.dataset.LocalOCRBenchDataset = LocalOCRBenchDataset
    vlmeval.dataset.LocalTreeBenchDataset = LocalTreeBenchDataset
    print("[run_vlmevalkit_local] Registered ImagineModel and local dataset adapters")


def step4_run_vlmevalkit(config_path, args, display_name, vlmeval_name):
    """Call VLMEvalKit's main logic."""
    work_dir = args.work_dir
    if work_dir is None:
        work_dir = os.path.join(
            PROJECT_ROOT, "results", f"vlmevalkit_{vlmeval_name}_{display_name}"
        )

    # Build sys.argv for VLMEvalKit's arg parser
    sys.argv = [
        "run.py",
        "--config", config_path,
        "--work-dir", work_dir,
        "--mode", args.mode,
        "--api-nproc", str(args.api_nproc),
    ]
    if args.reuse:
        sys.argv.append("--reuse")
    if args.judge:
        sys.argv.extend(["--judge", args.judge])
    if args.judge_args:
        sys.argv.extend(["--judge-args", args.judge_args])

    print(f"\n{'='*60}")
    print(f"  VLMEvalKit Local Evaluation")
    print(f"{'='*60}")
    print(f"  Dataset:      {args.dataset} -> {vlmeval_name}")
    print(f"  Model:        {display_name}")
    print(f"  Prompt:       {args.prompt}")
    print(f"  Sandbox:      {args.exe_code}")
    print(f"  Judge:        {args.judge}")
    print(f"  Judge args:   {args.judge_args}")
    print(f"  Eval method:  {args.eval_method}")
    if args.eval_method == "llm_judge":
        print(f"  LLM judge model: {args.llm_judge_model}")
        print(f"  LLM judge config: {resolve(args.llm_judge_api_config)}")
    print(f"  Work dir:     {work_dir}")
    print(f"  Config:       {config_path}")
    print(f"{'='*60}\n")

    # Import and run VLMEvalKit main
    vlmeval_run = os.path.join(VLMEVAL_ROOT, "run.py")
    assert os.path.exists(vlmeval_run), f"VLMEvalKit run.py not found at {vlmeval_run}"

    # Use runpy to execute run.py in the current process
    import runpy
    old_argv = sys.argv
    try:
        runpy.run_path(vlmeval_run, run_name="__main__")
    finally:
        sys.argv = old_argv
    return work_dir


def main():
    args = parse_args()

    print(f"\n[Step 1/4] Converting local data to VLMEvalKit TSV ...")
    tsv_path, vlmeval_name = step1_convert_data(
        args.dataset,
        resolve(args.data_path),
        resolve(args.image_dir),
        ocrbench_include_original=args.ocrbench_include_original,
        ocrbench_original_data_path=args.ocrbench_original_data_path,
    )

    print(f"[Step 2/4] Generating VLMEvalKit config ...")
    config_path, display_name = step2_generate_config(args, vlmeval_name, tsv_path)

    print(f"[Step 3/4] Registering ImagineModel into VLMEvalKit ...")
    step3_register_components()

    print(f"[Step 4/4] Running VLMEvalKit ...")
    work_dir = step4_run_vlmevalkit(config_path, args, display_name, vlmeval_name)
    _ensure_infer_result_answers(args, work_dir, display_name, vlmeval_name)
    _report_infer_accuracy_from_extra_records(args, work_dir, display_name, vlmeval_name)
    _report_cv_bench_summary_from_extra_records(args, work_dir, display_name, vlmeval_name)


if __name__ == "__main__":
    main()
