"""
Preprocess the V*Bench dataset for EasyR1 validation.

V*Bench is a visual QA benchmark with multiple-choice questions about
visual attributes and spatial relationships.

Source: lmms-lab/vstar-bench (HuggingFace)

Usage:
    export HF_ENDPOINT=https://hf-mirror.com  # if needed
    python examples/data_preprocess/vstar_bench.py --save_dir /path/to/output
"""



import argparse
import json
import os

import datasets


def load_local_vstar_export(dataset_path: str) -> datasets.DatasetDict:
    """Load a locally exported V*Bench directory with data.json and images/."""
    dataset_path = os.path.abspath(os.path.expanduser(dataset_path))

    if os.path.isdir(dataset_path):
        data_json_path = os.path.join(dataset_path, "data.json")
        images_dir = os.path.join(dataset_path, "images")
    else:
        data_json_path = dataset_path
        images_dir = os.path.join(os.path.dirname(dataset_path), "images")

    if not os.path.exists(data_json_path):
        raise FileNotFoundError(f"Local V*Bench export not found: {data_json_path}")

    with open(data_json_path, "r", encoding="utf-8") as f:
        raw_rows = json.load(f)

    rows = []
    for row in raw_rows:
        image_name = row["image"]
        image_path = image_name if os.path.isabs(image_name) else os.path.join(images_dir, image_name)
        rows.append(
            {
                "text": row["text"],
                "label": row["label"],
                "image": image_path,
                "category": row.get("category"),
                "question_id": row.get("question_id"),
            }
        )

    test_dataset = datasets.Dataset.from_list(rows)
    test_dataset = test_dataset.cast_column("image", datasets.Image())
    return datasets.DatasetDict({"test": test_dataset})


def load_vstar_dataset(dataset_path: str | None) -> datasets.DatasetDict:
    data_source = "lmms-lab/vstar-bench"
    if dataset_path is None:
        return datasets.load_dataset(data_source)

    expanded_path = os.path.abspath(os.path.expanduser(dataset_path))
    if os.path.isdir(expanded_path) and os.path.exists(os.path.join(expanded_path, "data.json")):
        return load_local_vstar_export(expanded_path)
    if expanded_path.endswith(".json") and os.path.exists(expanded_path):
        return load_local_vstar_export(expanded_path)

    return datasets.load_dataset(dataset_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        default="data/vstar_bench",
        help="Directory to save the preprocessed parquet files.",
    )
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="Optional local dataset path. Supports a Hugging Face dataset path, a local export directory with data.json and images/, or a local data.json file.",
    )
    args = parser.parse_args()

    dataset = load_vstar_dataset(args.dataset_path)

    test_dataset = dataset["test"]

    def process_fn(example):
        # EasyR1 expects: problem (str), answer (str), images (list of PIL images)
        # V*Bench has: text (question+options), label (A/B/C/D), image (PIL Image)
        example["problem"] = "<image>" + example["text"]
        example["answer"] = example["label"]
        example["images"] = [example["image"]]
        return example

    test_processed = test_dataset.map(
        function=process_fn,
        remove_columns=["text", "label", "image", "category", "question_id"],
        num_proc=1,
    )

    save_dir = os.path.expanduser(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    test_processed.to_parquet(os.path.join(save_dir, "test.parquet"))
    print(f"Saved test ({len(test_processed)}) to {save_dir}")


if __name__ == "__main__":
    main()
