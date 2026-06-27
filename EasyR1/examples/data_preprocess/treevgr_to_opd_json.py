"""
Sample TreeVGR records and convert them to the normalized OPD JSON schema.

The generated JSON can be passed to examples/data_preprocess/opd_json.py to
create the EasyR1 train.parquet file.

"""

import argparse
import json
import random
from pathlib import Path
from typing import Any


def resolve_path(path: str, root: Path) -> str:
    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        expanded = root / expanded
    return str(expanded.resolve())


def build_record(item: dict[str, Any], source_path: Path, source_pos: int, root: Path, seed: int) -> dict[str, Any]:
    source_index = item.get("index", source_pos)
    student_images = [resolve_path(path, root) for path in item.get("images") or []]
    teacher_extra = [resolve_path(path, root) for path in item.get("crop_images") or []]
    answer = str(item.get("answer") or "").strip()
    problem = str(item.get("problem") or "").strip()

    return {
        "sample_id": f"treevgr_seed{seed}_{source_index}",
        "source_json": str(source_path),
        "source_index": source_index,
        "op_type": "zoom",
        "messages": [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": f"<answer>{answer}</answer>"},
        ],
        "images": student_images,
        # opd_json.py expects originals first and strips them before writing parquet.
        "teacher_images": student_images + teacher_extra,
        "target_answer": answer,
        "target_instances": item.get("target_instances") or [],
    }


def validate_image_paths(records: list[dict[str, Any]]) -> None:
    missing = []
    for record in records:
        for key in ("images", "teacher_images"):
            for path in record.get(key) or []:
                if not Path(path).exists():
                    missing.append(path)

    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} referenced image(s) are missing. First ones:\n{preview}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to TreeVGR data.json.")
    parser.add_argument("--save_path", required=True, help="Path to normalized OPD JSON output.")
    parser.add_argument("--raw_save_path", default=None, help="Optional path to save sampled raw TreeVGR records.")
    parser.add_argument("--sample_size", type=int, default=6000, help="Number of samples to draw.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--root",
        default=None,
        help="Dataset root used to resolve relative image paths. Defaults to the input JSON directory.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    root = Path(args.root).expanduser().resolve() if args.root else input_path.parent

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if args.sample_size > len(data):
        raise ValueError(f"sample_size={args.sample_size} exceeds dataset size {len(data)}")

    rng = random.Random(args.seed)
    sampled_positions = rng.sample(range(len(data)), args.sample_size)
    sampled = [data[pos] for pos in sampled_positions]

    if args.raw_save_path:
        raw_save_path = Path(args.raw_save_path).expanduser().resolve()
        raw_save_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_save_path.open("w", encoding="utf-8") as f:
            json.dump(sampled, f, ensure_ascii=False, indent=2)
            f.write("\n")

    normalized = [
        build_record(item, source_path=input_path, source_pos=source_pos, root=root, seed=args.seed)
        for item, source_pos in zip(sampled, sampled_positions)
    ]
    validate_image_paths(normalized)

    save_path = Path(args.save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Input records: {len(data)}")
    print(f"Sampled records: {len(sampled)}")
    print(f"Seed: {args.seed}")
    if args.raw_save_path:
        print(f"Saved raw sample to: {Path(args.raw_save_path).expanduser().resolve()}")
    print(f"Saved normalized OPD JSON to: {save_path}")


if __name__ == "__main__":
    main()
