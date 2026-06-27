#!/usr/bin/env python3
"""
eval/run_eval.py

Generic evaluation entrypoint for switching datasets, prompts, and models.
Dataset-specific logic lives in eval/eval_<dataset>.py modules.
"""

import json
import os
import re
import sys
import time
import argparse
import traceback
import importlib

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from openai import OpenAI
from inference_engine.vis_inference_demo_gpt import evaluate_single_with_cleanup

# ==================== Dataset Module Registry ====================

DATASET_MODULES = {
    "vstar": "eval.eval_vstar",
    "thyme": "eval.eval_thyme",
    "monet": "eval.eval_monet",
    "hr_bench": "eval.eval_hr_bench",
    "ocrbench": "eval.eval_ocrbench",
}


def get_dataset_module(dataset_name):
    """Import and return the dataset-specific evaluation module."""
    if dataset_name not in DATASET_MODULES:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_MODULES.keys())}"
        )
    return importlib.import_module(DATASET_MODULES[dataset_name])


def get_item_image_paths(item):
    image_path_list = item.get("image_path_list")
    if isinstance(image_path_list, list) and image_path_list:
        return [path for path in image_path_list if path]

    image_path = item.get("image_path")
    return [image_path] if image_path else []


def image_paths_exist(image_paths):
    return bool(image_paths) and all(os.path.exists(path) for path in image_paths)


def get_cached_image_paths(result):
    image_path_list = result.get("image_path_list")
    if isinstance(image_path_list, list) and image_path_list:
        return [path for path in image_path_list if path]

    image_path = result.get("image_path")
    return [image_path] if image_path else []


# ==================== Mode Analysis ====================

def extract_all_assistant_text(messages, final_response):
    """Extract all assistant-generated text from messages + final_response."""
    texts = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
        elif isinstance(content, str):
            texts.append(content)
    if isinstance(final_response, str):
        texts.append(final_response)
    return "\n".join(texts)


def analyze_mode(messages, final_response):
    """Analyze which reasoning mode was used in the response."""
    all_text = extract_all_assistant_text(messages, final_response)
    if not all_text:
        return {
            "has_think_image": False, "has_code": False,
            "think_image_count": 0, "code_count": 0,
            "mode": "none"
        }
    think_image_count = len(re.findall(r'<think_image>', all_text))
    code_count = len(re.findall(r'<code>', all_text))
    has_think_image = think_image_count > 0
    has_code = code_count > 0

    if has_think_image and not has_code:
        mode = "text_imagine"
    elif has_code and not has_think_image:
        mode = "code"
    elif has_think_image and has_code:
        mode = "mixed"
    else:
        mode = "direct"

    return {
        "has_think_image": has_think_image, "has_code": has_code,
        "think_image_count": think_image_count, "code_count": code_count,
        "mode": mode
    }


JUDGE_SYSTEM_PROMPT = """You are an impartial evaluation judge.
Your task is to determine whether a model prediction should be considered correct given:
- the question
- the ground-truth answer
- the model's extracted answer
- the model's raw response

Judge based on the task type implied by the question.

1. OCR / text transcription / text extraction tasks:
- Be very strict.
- If the question asks to identify, read, transcribe, or return the full line/text from the image,
  the prediction must match the ground-truth text essentially exactly.
- Extra words, missing words, added units, changed characters, altered numbers, or completing a truncated ground truth
  should normally be marked incorrect.
- Minor whitespace-only differences may be ignored.

2. Numeric / chart / reasoning tasks:
- Focus on whether the final answer is factually correct.
- Be tolerant to harmless formatting differences such as:
  capitalization, punctuation, surrounding answer phrases, equivalent numeric forms like 1999 vs 1999.0,
  and concise paraphrases that preserve the same meaning.
- Do not accept answers that change magnitude or units, such as 2.63 vs 2.63k, unless the question explicitly expects that unit.

3. If the extracted answer is contaminated by long reasoning text:
- Use the model_raw_response only to understand the intended final answer.
- Do not automatically mark it correct just because the raw response contains the ground-truth somewhere.
- For strict OCR-style tasks, answer text pollution should usually be marked incorrect.
- For reasoning tasks, if the intended final answer is still clear and correct, it may be marked correct.

Be strict about factual correctness and follow the stricter interpretation when uncertain.

Return only valid JSON with this schema:
{
  "is_correct": true or false,
  "score": 1 or 0,
  "verdict": "correct" or "incorrect",
  "reason": "short explanation"
}
"""


def build_eval_record(name, pred_answer, is_correct, extra=None):
    record = {
        "name": name,
        "pred_answer": pred_answer,
        "is_correct": bool(is_correct),
    }
    if extra:
        record.update(extra)
    return record


def init_eval_stats():
    return {
        "total": 0,
        "correct": 0,
        "category_stats": {},
        "split_category_stats": {},
    }


def update_eval_stats(stats, category_name, split_name, is_correct_value):
    stats["total"] += 1
    if is_correct_value:
        stats["correct"] += 1

    category_name = category_name or "unknown"
    split_name = split_name or "unknown"

    if category_name not in stats["category_stats"]:
        stats["category_stats"][category_name] = {"total": 0, "correct": 0}
    stats["category_stats"][category_name]["total"] += 1
    if is_correct_value:
        stats["category_stats"][category_name]["correct"] += 1

    if split_name not in stats["split_category_stats"]:
        stats["split_category_stats"][split_name] = {}
    if category_name not in stats["split_category_stats"][split_name]:
        stats["split_category_stats"][split_name][category_name] = {"total": 0, "correct": 0}
    stats["split_category_stats"][split_name][category_name]["total"] += 1
    if is_correct_value:
        stats["split_category_stats"][split_name][category_name]["correct"] += 1


def print_eval_summary_block(title, stats):
    total = stats["total"]
    correct = stats["correct"]
    accuracy = (correct / total * 100) if total > 0 else 0

    print(f"\n  {title}:")
    print(f"    Total:    {total}")
    print(f"    Correct:  {correct}")
    print(f"    Accuracy: {accuracy:.2f}%")

    print(f"\n    Category Breakdown:")
    for cat_name in sorted(stats["category_stats"].keys()):
        cs = stats["category_stats"][cat_name]
        cat_acc = cs["correct"] / cs["total"] * 100 if cs["total"] > 0 else 0
        print(f"      {cat_name.upper()}: {cs['correct']}/{cs['total']} = {cat_acc:.2f}%")

    if stats["split_category_stats"]:
        print(f"\n    Split Breakdown:")
        for split_name in sorted(stats["split_category_stats"].keys()):
            split_stats = stats["split_category_stats"][split_name]
            split_total = sum(v["total"] for v in split_stats.values())
            split_correct = sum(v["correct"] for v in split_stats.values())
            split_acc = split_correct / split_total * 100 if split_total > 0 else 0
            print(f"      {split_name.upper()}: {split_correct}/{split_total} = {split_acc:.2f}%")
            for cat_name in sorted(split_stats.keys()):
                cs = split_stats[cat_name]
                cat_acc = cs["correct"] / cs["total"] * 100 if cs["total"] > 0 else 0
                print(f"        {cat_name.upper()}: {cs['correct']}/{cs['total']} = {cat_acc:.2f}%")


def build_eval_summary_dict(stats):
    total = stats["total"]
    correct = stats["correct"]
    accuracy = (correct / total * 100) if total > 0 else 0
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "per_category": {
            cat: {
                "total": cs["total"],
                "correct": cs["correct"],
                "accuracy": round(cs["correct"] / cs["total"] * 100, 2) if cs["total"] > 0 else 0,
            }
            for cat, cs in stats["category_stats"].items()
        },
        "per_split": {
            split_name: {
                "total": sum(v["total"] for v in split_stats.values()),
                "correct": sum(v["correct"] for v in split_stats.values()),
                "accuracy": round(
                    sum(v["correct"] for v in split_stats.values()) /
                    sum(v["total"] for v in split_stats.values()) * 100,
                    2,
                ) if sum(v["total"] for v in split_stats.values()) > 0 else 0,
                "per_category": {
                    cat: {
                        "total": cs["total"],
                        "correct": cs["correct"],
                        "accuracy": round(cs["correct"] / cs["total"] * 100, 2) if cs["total"] > 0 else 0,
                    }
                    for cat, cs in split_stats.items()
                },
            }
            for split_name, split_stats in stats["split_category_stats"].items()
        },
    }


def parse_judge_json(content):
    if not content:
        raise ValueError("Empty judge response")
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def judge_with_llm(judge_client, judge_model, item, extracted_answer, raw_final_response):
    user_payload = {
        "question": item.get("question"),
        "ground_truth_answer": item.get("gt_label"),
        "model_extracted_answer": extracted_answer,
        "model_raw_response": raw_final_response,
        "category": item.get("category"),
        "split": item.get("split", "unknown"),
    }
    response = judge_client.chat.completions.create(
        model=judge_model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ],
    )
    content = response.choices[0].message.content if response.choices else ""
    parsed = parse_judge_json(content)
    is_correct = bool(parsed.get("is_correct", False))
    return {
        "name": "llm_judge_eval",
        "pred_answer": extracted_answer,
        "is_correct": is_correct,
        "score": int(parsed.get("score", 1 if is_correct else 0)),
        "verdict": parsed.get("verdict", "correct" if is_correct else "incorrect"),
        "reason": parsed.get("reason", ""),
        "judge_model": judge_model,
        "judge_raw_response": content,
    }




# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Universal evaluation script")
    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to data JSON file")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_MODULES.keys()),
                        help="Dataset name (determines loader, answer extraction, and correctness check)")
    parser.add_argument(
        "--ocrbench_include_original",
        action="store_true",
        help="For OCRBench transformed variants, also provide the matching original image.",
    )
    parser.add_argument(
        "--ocrbench_original_data_path",
        type=str,
        default=None,
        help="Optional path to OCRBench original data.json used to resolve the matching original image.",
    )
    # Model
    parser.add_argument("--api_config", type=str, required=True, help="Path to API config JSON")
    parser.add_argument("--model_name", type=str, default="Thyme-RL", help="Model name for API")
    parser.add_argument("--client_type", type=str, default="openai", choices=["openai", "vllm", "anthropic"])
    # Prompt
    parser.add_argument("--prompt_template", type=str, required=True, help="Path to prompt template JSON")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt key in template JSON")
    # Inference
    parser.add_argument("--exe_code", action="store_true", help="Enable code execution")
    parser.add_argument("--max_tokens", type=int, default=16000)
    parser.add_argument("--max_rounds", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    # Output
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (auto-generated if not set)")
    # Range
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    # Judge
    parser.add_argument("--enable_llm_judge", action="store_true",
                        help="Enable an additional LLM-as-judge evaluation pass")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--judge_key", type=str,
                        default="xxx")
    parser.add_argument("--judge_api_base", type=str, default="xxx")

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = current_dir
    for attr in [
        "data_path",
        "image_dir",
        "api_config",
        "prompt_template",
        "output_dir",
        "ocrbench_original_data_path",
    ]:
        val = getattr(args, attr)
        if val and not os.path.isabs(val):
            setattr(args, attr, os.path.join(project_root, val))

    # Auto output dir
    if args.output_dir is None:
        data_name = os.path.basename(os.path.dirname(args.data_path))
        args.output_dir = os.path.join(project_root, "results", f"eval_{data_name}_{args.model_name}_{args.prompt}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load API client
    with open(args.api_config) as f:
        api_cfg = json.load(f)
    client = OpenAI(api_key=api_cfg["api_key"][0], base_url=api_cfg.get("base_url"))
    judge_client = None
    if args.enable_llm_judge:
        judge_client = OpenAI(api_key=args.judge_key, base_url=args.judge_api_base)

    # Load dataset-specific module
    ds_mod = get_dataset_module(args.dataset)
    load_kwargs = {}
    if args.dataset == "ocrbench":
        load_kwargs = {
            "include_original_image": args.ocrbench_include_original,
            "original_data_path": args.ocrbench_original_data_path,
        }
    items = ds_mod.load_data(args.data_path, args.image_dir, **load_kwargs)

    # Slice
    end_idx = args.end_idx if args.end_idx > 0 else len(items)
    items = items[args.start_idx:end_idx]

    print(f"{'='*60}")
    print(f"  Evaluation Configuration")
    print(f"{'='*60}")
    print(f"  Data:       {args.data_path} ({len(items)} items)")
    print(f"  Model:      {args.model_name}")
    print(f"  Prompt:     {args.prompt}")
    print(f"  Exe Code:   {args.exe_code}")
    print(f"  Output:     {args.output_dir}")
    print(f"{'='*60}\n")

    # Eval args for inference engine
    eval_args = {
        "max_tokens": args.max_tokens,
        "prompt_template": args.prompt_template,
        "prompt": args.prompt,
        "exe_code": args.exe_code,
        "temperature": args.temperature,
        "max_rounds": args.max_rounds,
        "client_type": args.client_type,
        "api_name": args.model_name,
    }

    # Statistics
    rule_stats = init_eval_stats()
    judge_stats = init_eval_stats()
    total_input_tokens = 0
    total_output_tokens = 0
    mode_stats = {"text_imagine": 0, "code": 0, "mixed": 0, "direct": 0, "none": 0}

    for i, item in enumerate(items):
        global_idx = args.start_idx + i
        q_id = item["question_id"]
        safe_id = "".join(x for x in str(q_id) if x.isalnum() or x in "._-")
        result_file = os.path.join(args.output_dir, f"result_{global_idx}_{safe_id}.json")

        current_image_paths = get_item_image_paths(item)
        if not image_paths_exist(current_image_paths):
            print(f"[{global_idx+1}] Skip: image not found ({current_image_paths})")
            continue

        # Check cached result
        if os.path.exists(result_file):
            try:
                with open(result_file) as f:
                    cached = json.load(f)
                if get_cached_image_paths(cached) != current_image_paths:
                    raise ValueError("Cached image inputs do not match the current evaluation item")
                if "raw_final_response" in cached:
                    cat = cached.get("category", "unknown")
                    split_name = cached.get("split", item.get("split", "unknown"))
                    rule_eval = cached.get("rule_eval")
                    if rule_eval is None:
                        legacy_is_correct = cached.get("is_correct", False)
                        rule_eval = build_eval_record(
                            "rule_eval",
                            cached.get("pred_answer"),
                            legacy_is_correct,
                        )
                    update_eval_stats(rule_stats, cat, split_name, rule_eval.get("is_correct", False))

                    judge_eval = cached.get("llm_judge_eval")
                    if args.enable_llm_judge and judge_eval and judge_eval.get("available", True):
                        update_eval_stats(judge_stats, cat, split_name, judge_eval.get("is_correct", False))

                    total_input_tokens += cached.get("input_tokens", 0)
                    total_output_tokens += cached.get("output_tokens", 0)
                    mode = cached.get("mode_analysis", {}).get("mode", "none")
                    mode_stats[mode] = mode_stats.get(mode, 0) + 1
                    print(
                        f"[{global_idx+1}/{args.start_idx+len(items)}] Cached | "
                        f"Rule={rule_eval.get('is_correct', False)}"
                        + (
                            f" | Judge={judge_eval.get('is_correct', False)}"
                            if args.enable_llm_judge and judge_eval
                            else ""
                        )
                        + f" | {cat}"
                    )
                    continue
            except Exception:
                pass

        print(f"\n[{global_idx+1}/{args.start_idx+len(items)}] ID={q_id} cat={item['category']}")

        # Run inference
        data_input = {"question": item["question"], "image_path_list": current_image_paths}
        try:
            messages, final_response, in_tok, out_tok = evaluate_single_with_cleanup(
                eval_args, data_input, client
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            continue

        extracted_answer = ds_mod.extract_answer(final_response)
        if args.dataset == "ocrbench":
            rule_is_correct = ds_mod.check_correct(
                extracted_answer,
                item["gt_label"],
                item.get("category"),
            )
        else:
            rule_is_correct = ds_mod.check_correct(extracted_answer, item["gt_label"])
        rule_eval = build_eval_record(
            "rule_eval",
            extracted_answer,
            rule_is_correct,
        )

        judge_eval = None
        if args.enable_llm_judge:
            try:
                judge_eval = judge_with_llm(
                    judge_client=judge_client,
                    judge_model=args.judge_model,
                    item=item,
                    extracted_answer=extracted_answer,
                    raw_final_response=final_response,
                )
            except Exception as judge_err:
                judge_eval = {
                    "name": "llm_judge_eval",
                    "pred_answer": extracted_answer,
                    "is_correct": False,
                    "available": False,
                    "error": str(judge_err),
                }
        mode_analysis = analyze_mode(messages, final_response)

        cat = item["category"]
        split_name = item.get("split", "unknown")
        update_eval_stats(rule_stats, cat, split_name, rule_eval["is_correct"])
        if args.enable_llm_judge and judge_eval.get("available", True):
            update_eval_stats(judge_stats, cat, split_name, judge_eval["is_correct"])

        total_input_tokens += in_tok
        total_output_tokens += out_tok
        mode_stats[mode_analysis["mode"]] = mode_stats.get(mode_analysis["mode"], 0) + 1

        print(
            f"  GT={item['gt_label']} | Pred={extracted_answer} | "
            f"Rule={rule_eval['is_correct']}"
            + (
                f" | Judge={judge_eval.get('is_correct', False)}"
                if args.enable_llm_judge
                else ""
            )
            + f" | Mode={mode_analysis['mode']} | Tokens: in={in_tok} out={out_tok}"
        )
        print(f"  rule_eval={json.dumps(rule_eval, ensure_ascii=False)}")
        if args.enable_llm_judge:
            print(f"  llm_judge_eval={json.dumps(judge_eval, ensure_ascii=False)}")

        # Save result
        result_item = {
            "data_idx": global_idx,
            "question_id": q_id,
            "question": item["question"],
            "image_path": current_image_paths[0],
            "image_path_list": current_image_paths,
            "gt_label": item["gt_label"],
            "split": split_name,
            "category": cat,
            "pred_answer": extracted_answer,
            "is_correct": rule_eval["is_correct"],
            "rule_eval": rule_eval,
            "llm_judge_eval": judge_eval,
            "mode_analysis": mode_analysis,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "raw_final_response": final_response,
            "messages": messages,
        }
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_item, f, indent=2, ensure_ascii=False)

        time.sleep(0.2)

    # ==================== Summary ====================
    total = rule_stats["total"]
    avg_in_tok = total_input_tokens / total if total > 0 else 0
    avg_out_tok = total_output_tokens / total if total > 0 else 0
    avg_total_tok = avg_in_tok + avg_out_tok

    print(f"\n{'='*60}")
    print(f"  Evaluation Summary")
    print(f"{'='*60}")
    print(f"  Model:          {args.model_name}")
    print(f"  Prompt:         {args.prompt}")
    print(f"  Data:           {args.data_path}")
    print(f"  Total:          {total}")
    print(f"{'='*60}")
    print_eval_summary_block("Rule Eval", rule_stats)
    if args.enable_llm_judge:
        print_eval_summary_block("LLM Judge Eval", judge_stats)

    # Token stats
    print(f"\n  Token Consumption:")
    print(f"    Avg Input Tokens:  {avg_in_tok:.1f}")
    print(f"    Avg Output Tokens: {avg_out_tok:.1f}")
    print(f"    Avg Total Tokens:  {avg_total_tok:.1f}")
    print(f"    Total Input Tokens:  {total_input_tokens}")
    print(f"    Total Output Tokens: {total_output_tokens}")

    # Mode distribution
    print(f"\n  Mode Distribution:")
    for mode_name, count in sorted(mode_stats.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = count / total * 100 if total > 0 else 0
            print(f"    {mode_name}: {count} ({pct:.1f}%)")

    print(f"{'='*60}")

    # Save summary
    summary_filename = f"{args.model_name}_summary.json"
    summary = {
        "config": {
            "model_name": args.model_name,
            "prompt": args.prompt,
            "data_path": args.data_path,
            "exe_code": args.exe_code,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "enable_llm_judge": args.enable_llm_judge,
            "judge_model": args.judge_model if args.enable_llm_judge else None,
            "judge_api_base": args.judge_api_base if args.enable_llm_judge else None,
        },
        "rule_eval": build_eval_summary_dict(rule_stats),
        "llm_judge_eval": build_eval_summary_dict(judge_stats) if args.enable_llm_judge else None,
        "token_stats": {
            "avg_input_tokens": round(avg_in_tok, 1),
            "avg_output_tokens": round(avg_out_tok, 1),
            "avg_total_tokens": round(avg_total_tok, 1),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        },
        "mode_distribution": {k: v for k, v in mode_stats.items() if v > 0},
    }
    summary_path = os.path.join(args.output_dir, summary_filename)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
