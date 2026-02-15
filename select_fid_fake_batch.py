#!/usr/bin/env python3
"""Create balanced random fake-image subsets from multiple generated folders."""

import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    """Parse command-line arguments for batch fake-image subset selection."""
    p = argparse.ArgumentParser(
        description=(
            "Batch-select balanced random subsets from multiple generated_images_* folders "
            "and copy into separate flat destination folders."
        )
    )
    p.add_argument(
        "--src-base",
        default="stylegan2-ada-pytorch",
        help="Base directory containing generated_images_* folders.",
    )
    p.add_argument(
        "--src-glob",
        default="generated_images_*",
        help="Glob pattern for source folders.",
    )
    p.add_argument(
        "--dst-base",
        default="tmp",
        help="Base directory for output folders.",
    )
    p.add_argument(
        "--dst-prefix",
        default="fid_fake_flat_",
        help="Prefix for output folder names.",
    )
    p.add_argument(
        "--total",
        type=int,
        default=1000,
        help="Total images to copy per source folder.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (omit for non-deterministic selection).",
    )
    return p.parse_args()


def select_one(src_root: Path, dst_root: Path, total: int, seed: int | None):
    """Select a balanced random subset from one source tree and copy it flat.

    The source directory must contain one subdirectory per class. Files are
    sampled approximately evenly across classes and copied into a single output
    directory using `<class>__<filename>` naming to avoid collisions.
    """
    if seed is not None:
        random.seed(seed)
    else:
        random.seed()

    classes = sorted([p for p in src_root.iterdir() if p.is_dir()])
    if not classes:
        raise SystemExit(f"No class subdirectories found in {src_root}")

    n_classes = len(classes)
    if total < n_classes:
        raise SystemExit("Total must be >= number of classes.")

    per_class = total // n_classes
    remainder = total % n_classes

    class_order = classes[:]
    random.shuffle(class_order)

    if dst_root.exists():
        raise SystemExit(f"Destination already exists: {dst_root}")

    dst_root.mkdir(parents=True, exist_ok=False)

    selected_total = 0
    for i, cls in enumerate(class_order):
        files = [p for p in cls.iterdir() if p.is_file()]
        k = per_class + (1 if i < remainder else 0)
        if len(files) < k:
            raise SystemExit(f"Not enough files in {cls.name}: {len(files)}")
        picks = random.sample(files, k)
        for p in picks:
            dst_name = f"{cls.name}__{p.name}"
            shutil.copy2(p, dst_root / dst_name)
        selected_total += k

    print(f"Copied {selected_total} files into {dst_root}")


def main():
    """Run selection for all source folders matching the configured glob."""
    args = parse_args()

    src_base = Path(args.src_base)
    dst_base = Path(args.dst_base)

    src_roots = sorted([p for p in src_base.glob(args.src_glob) if p.is_dir()])
    if not src_roots:
        raise SystemExit(f"No source folders matched {args.src_glob} in {src_base}")

    for src_root in src_roots:
        suffix = src_root.name.replace("generated_images_", "")
        dst_root = dst_base / f"{args.dst_prefix}{suffix}"
        select_one(src_root, dst_root, args.total, args.seed)


if __name__ == "__main__":
    main()
