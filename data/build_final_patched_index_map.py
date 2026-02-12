"""Create an index-to-filename map for final_patched_BTXRD image ordering.

The resulting JSON is used to translate split indices from ``dataset_split.json``
into concrete image filenames for downstream filtering steps.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

# Default paths relative to the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_PATH = REPO_ROOT / "data" / "dataset" / "dataset_split.json"
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "dataset" / "final_patched_BTXRD"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "dataset" / "final_patched_index_map.json"


def load_split_indices(split_path: Path) -> List[int]:
    """Load and merge train/test split indices from dataset_split.json."""
    split_data = json.loads(split_path.read_text())
    try:
        train_indices = split_data["train"]
        test_indices = split_data["test"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{split_path} must contain 'train' and 'test' lists") from exc

    if not isinstance(train_indices, list) or not isinstance(test_indices, list):
        raise ValueError(f"'train' and 'test' in {split_path} must be lists of integers.")

    try:
        return [int(v) for v in train_indices] + [int(v) for v in test_indices]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'train' and 'test' in {split_path} must be lists of integers.") from exc


def list_images(dataset_dir: Path) -> List[str]:
    """List .jpeg filenames in deterministic lexicographic order."""
    return sorted([path.name for path in dataset_dir.glob("*.jpeg")])


def validate_indices(indices: Sequence[int], total_images: int) -> None:
    """Validate that split indices form a full zero-based permutation."""
    if not indices:
        raise ValueError("Split indices are empty.")

    unique_indices = set(indices)
    if len(unique_indices) != len(indices):
        raise ValueError("Split indices contain duplicates.")

    min_index = min(unique_indices)
    max_index = max(unique_indices)
    if min_index != 0 or max_index != total_images - 1:
        raise ValueError(
            f"Split indices must cover 0-{total_images - 1}, got {min_index}-{max_index}."
        )

    if len(unique_indices) != total_images:
        raise ValueError(
            f"Split contains {len(unique_indices)} indices but dataset has {total_images} images."
        )


def build_index_map(images: Sequence[str]) -> Dict[str, str]:
    """Build JSON-serializable mapping: string index -> image filename."""
    return {str(idx): name for idx, name in enumerate(images)}


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a JSON dictionary mapping index -> IMG filename based on final_patched_BTXRD ordering."
        )
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help=f"Path to dataset_split.json (default: {DEFAULT_SPLIT_PATH})",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Directory with final_patched_BTXRD images (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    """Run validation and write the final index map JSON to disk."""
    args = parse_args()
    # Ordering must be stable so index assignments stay reproducible.
    images = list_images(args.dataset_dir)
    if not images:
        raise SystemExit(f"No .jpeg files found in {args.dataset_dir}")

    indices = load_split_indices(args.split_path)
    # Guard against mismatches between split metadata and actual dataset content.
    validate_indices(indices, total_images=len(images))

    index_map = build_index_map(images)
    args.output_path.write_text(json.dumps(index_map, indent=2))

    print(f"Wrote {len(index_map)} entries to {args.output_path}")


if __name__ == "__main__":
    main()
