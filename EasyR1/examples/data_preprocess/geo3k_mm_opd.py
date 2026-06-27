"""
Preprocess the Geometry3k dataset for multimodal OPD on EasyR1.
Adds teacher_images (random extra image per sample) for context distillation.

Usage:
    python examples/data_preprocess/geo3k_mm_opd.py \
        --save_dir /path/to/output
"""

import argparse
import os
import random

import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        default="data/geo3k_mm_opd",
        help="Directory to save the preprocessed parquet files.",
    )
    parser.add_argument("--dataset_path", default=None, help="Local path to the raw dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for teacher image selection.")
    args = parser.parse_args()

    data_source = "hiyouga/geometry3k"
    if args.dataset_path is not None:
        dataset = datasets.load_dataset(args.dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    def make_map_fn(split, ds):
        num_samples = len(ds)
        rng = random.Random(args.seed)
        # Pre-generate random indices for teacher extra images (different from current sample)
        teacher_indices = []
        for i in range(num_samples):
            idx = rng.randint(0, num_samples - 2)
            if idx >= i:
                idx += 1
            teacher_indices.append(idx)

        def process_fn(example, idx):
            # EasyR1 expects: problem (str), answer (str), images (list)
            # We add: teacher_images (list of extra images for teacher)
            teacher_img_idx = teacher_indices[idx]
            teacher_images = [ds[teacher_img_idx]["images"][0]]

            example["teacher_images"] = teacher_images
            return example

        return process_fn

    train_processed = train_dataset.map(
        function=make_map_fn("train", train_dataset), with_indices=True, num_proc=1
    )
    test_processed = test_dataset.map(
        function=make_map_fn("test", test_dataset), with_indices=True, num_proc=1
    )

    save_dir = os.path.expanduser(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    train_processed.to_parquet(os.path.join(save_dir, "train.parquet"))
    test_processed.to_parquet(os.path.join(save_dir, "test.parquet"))

    print(f"Saved train ({len(train_processed)}) and test ({len(test_processed)}) to {save_dir}")


if __name__ == "__main__":
    main()
