"""
Inspect the teacher/reference prompt used by OPD and optionally generate outputs.

This script mirrors the teacher prompt construction in
``verl/trainer/ref_input_utils.py``:

- original images come from the parquet ``images`` column
- teacher extra images come from ``teacher_images``
- the student system prompt is replaced by the teacher system prompt
- the ground-truth answer text is appended to the user message

"""

import argparse
import re
from pathlib import Path
from typing import Any

import datasets
import torch
from transformers import AutoProcessor


DEFAULT_TEACHER_SYSTEM_PROMPT = (
    "You are given two images. The first image is the diagram for the question. "
    "The second image is an additional reference image that is NOT directly related to the question. "
    "Focus on the first image to solve the problem."
)


def load_text_or_value(value: str | None, default: str) -> str:
    if not value:
        return default
    path = Path(value)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return value


def strip_example_format(prompt: str) -> str:
    """Remove the Example format block while keeping the Output Format rules."""
    return re.sub(
        r"\n*Example format:\n.*?(?=\nOutput Format)",
        "\n",
        prompt,
        flags=re.DOTALL,
    ).strip()


def content_string_to_list(content: str) -> list[dict[str, Any]]:
    content_list: list[dict[str, Any]] = []
    parts = content.split("<image>")
    for idx, part in enumerate(parts):
        if idx > 0:
            content_list.append({"type": "image"})
        if part:
            content_list.append({"type": "text", "text": part})
    return content_list


def build_teacher_messages(
    problem: str,
    answer: str,
    teacher_system_prompt: str,
    num_teacher_extra_images: int,
    answer_prefix: str,
) -> list[dict[str, Any]]:
    # This matches ref_input_utils.py: keep the user content, append extra image
    # slots, append the ground-truth text, then prepend the teacher system prompt.
    user_content = content_string_to_list(problem)
    for _ in range(num_teacher_extra_images):
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": f"{answer_prefix}{answer}"})

    return [
        {"role": "system", "content": teacher_system_prompt},
        {"role": "user", "content": user_content},
    ]


def load_model(model_path: str, device: str):
    try:
        from transformers import AutoModelForImageTextToText

        model_cls = AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq

        model_cls = AutoModelForVision2Seq

    model = model_cls.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.eval()
    if device != "auto":
        model.to(device)
    return model


def move_inputs_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True, help="Path to train.parquet.")
    parser.add_argument("--model-path", required=True, help="Path to Qwen3-VL teacher/reference model.")
    parser.add_argument("--teacher-system-prompt", default=None, help="Prompt file path or literal prompt text.")
    parser.add_argument("--num-cases", type=int, default=3, help="Number of examples to inspect.")
    parser.add_argument("--indices", default=None, help="Comma-separated row indices. Overrides --num-cases.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0", help="Generation device, e.g. cuda:0 or cpu.")
    parser.add_argument("--print-prompt", action="store_true", help="Print rendered chat template prompt.")
    parser.add_argument("--hide-messages", action="store_true", help="Do not print the raw messages list.")
    parser.add_argument(
        "--strip-example-format",
        action="store_true",
        help="Temporarily remove the 'Example format' block from the teacher system prompt.",
    )
    parser.add_argument("--generate", action="store_true", help="Load model and generate teacher outputs.")
    parser.add_argument(
        "--answer-prefix",
        default="Ground-truth answer(option letter):",
        help="Text prepended before the ground-truth answer. Defaults to the current training code behavior.",
    )
    args = parser.parse_args()

    ds = datasets.load_dataset("parquet", data_files=args.data_file, split="train")
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    else:
        indices = list(range(min(args.num_cases, len(ds))))

    teacher_system_prompt = load_text_or_value(args.teacher_system_prompt, DEFAULT_TEACHER_SYSTEM_PROMPT)
    if args.strip_example_format:
        teacher_system_prompt = strip_example_format(teacher_system_prompt)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    model = None
    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if args.generate:
        model = load_model(args.model_path, args.device)
        if args.device == "auto":
            device = next(model.parameters()).device

    for case_id, row_idx in enumerate(indices):
        row = ds[row_idx]
        original_images = list(row.get("images") or [])
        teacher_extra_images = list(row.get("teacher_images") or [])
        all_images = original_images + teacher_extra_images
        messages = build_teacher_messages(
            problem=row["problem"],
            answer=row["answer"],
            teacher_system_prompt=teacher_system_prompt,
            num_teacher_extra_images=len(teacher_extra_images),
            answer_prefix=args.answer_prefix,
        )
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        print("=" * 100)
        print(f"CASE {case_id} | row_index={row_idx}")
        print(f"answer={row['answer']}")
        print(f"original_images={len(original_images)} teacher_extra_images={len(teacher_extra_images)}")
        if not args.hide_messages:
            print("messages=")
            print(messages)
        if args.print_prompt or not args.generate:
            print("rendered_prompt=")
            print(prompt)

        if not args.generate:
            continue

        inputs = processor(text=[prompt], images=all_images if all_images else None, return_tensors="pt")
        inputs = move_inputs_to_device(inputs, device)
        do_sample = args.temperature > 0
        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs["temperature"] = args.temperature
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                **generate_kwargs,
            )
        prompt_len = inputs["input_ids"].shape[1]
        generated_text = processor.batch_decode(
            generated_ids[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        print("teacher_output=")
        print(generated_text)


if __name__ == "__main__":
    main()
