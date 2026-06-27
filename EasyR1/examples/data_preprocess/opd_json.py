"""
Convert normalized OPD JSON to EasyR1 parquet format and duplicate to desired count.

Supported normalized input schema per sample:
    {
        "messages": [...],
        "images": ["path/or/base64", ...],
        "teacher_images": ["original_path_or_base64", "extra_path_or_base64", ...]
    }

The images may be base64 strings or local image paths. teacher_images should contain
all original images first, followed by extra teacher-only images.
"""

import argparse
import base64
import binascii
import os
import re

import datasets


def load_image_ref(image_ref: str, base_dir: str) -> str | dict[str, bytes | None]:
    """Resolve an image reference to a local path or raw bytes for HF Image()."""
    if not isinstance(image_ref, str):
        raise TypeError(f"Unsupported image reference type: {type(image_ref).__name__}")

    if image_ref.startswith("data:"):
        return decode_base64_image(image_ref)

    candidate = os.path.expanduser(image_ref)
    if not os.path.isabs(candidate):
        candidate = os.path.abspath(os.path.join(base_dir, candidate))
    if os.path.exists(candidate):
        return candidate

    try:
        return decode_base64_image(image_ref)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"Image reference is neither an existing path nor valid base64: {image_ref}") from exc


def decode_base64_image(b64_str: str) -> dict[str, bytes | None]:
    """Decode a base64 image string into the dict format accepted by HF Image()."""
    image_path = None
    if b64_str.startswith("data:"):
        header, b64_str = b64_str.split(",", 1)
        match = re.search(r"data:([^;]+)", header)
        if match:
            mime_type = match.group(1)
            if "/" in mime_type:
                image_path = f"image.{mime_type.split('/', 1)[1]}"
    img_bytes = base64.b64decode(b64_str)
    return {"path": image_path, "bytes": img_bytes}


def extract_answer(messages: list[dict]) -> str:
    """Extract answer content from assistant message <answer>...</answer>.

    Existing OPD multiple-choice data still returns an uppercase A-D letter,
    while non-MCQ data such as TreeVGR keeps the full answer text.
    """
    for msg in messages:
        if msg.get("role") == "assistant":
            match = re.search(r"<answer>\s*(.*?)\s*</answer>", msg["content"], re.DOTALL | re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                if re.fullmatch(r"[A-Da-d]", answer):
                    return answer.upper()
                return answer
    return "A"  # default fallback for legacy malformed MCQ records


def extract_problem(messages: list[dict]) -> str:
    """Extract the user question from student messages."""
    for msg in messages:
        if msg.get("role") == "user":
            return msg["content"]
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSON file.")
    parser.add_argument("--save_dir", default="data/opd_json", help="Output directory.")
    parser.add_argument("--duplicate_to", type=int, default=48, help="Duplicate samples to this count.")
    args = parser.parse_args()

    import json

    with open(args.input, "r") as f:
        data = json.load(f)
    input_dir = os.path.dirname(os.path.abspath(args.input))

    rows = {"problem": [], "answer": [], "images": [], "teacher_images": []}

    for item in data:
        problem = extract_problem(item["messages"])
        answer = extract_answer(item["messages"])

        # Decode student images
        student_images = [load_image_ref(image_ref, base_dir=input_dir) for image_ref in item["images"]]

        # teacher_images in JSON: [original_1, ..., original_n, intermediate_1, ...]
        # Our pipeline expects teacher_images = EXTRA images only (not the original).
        teacher_refs = item.get("teacher_images", [])
        teacher_extra = [
            load_image_ref(image_ref, base_dir=input_dir) for image_ref in teacher_refs[len(student_images) :]
        ]

        rows["problem"].append(problem)
        rows["answer"].append(answer)
        rows["images"].append(student_images)
        rows["teacher_images"].append(teacher_extra)

    # Duplicate to desired count
    n_original = len(rows["problem"])
    if args.duplicate_to > n_original:
        repeats = args.duplicate_to // n_original
        remainder = args.duplicate_to % n_original
        for key in rows:
            rows[key] = rows[key] * repeats + rows[key][:remainder]

    print(f"Original: {n_original} samples, duplicated to: {len(rows['problem'])} samples")

    # Create HuggingFace dataset
    ds = datasets.Dataset.from_dict(rows)
    # Cast images to HF Image feature
    ds = ds.cast_column("images", datasets.Sequence(datasets.Image()))
    ds = ds.cast_column("teacher_images", datasets.Sequence(datasets.Image()))

    save_dir = os.path.expanduser(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    ds.to_parquet(os.path.join(save_dir, "train.parquet"))
    print(f"Saved {len(ds)} samples to {save_dir}/train.parquet")


if __name__ == "__main__":
    main()
