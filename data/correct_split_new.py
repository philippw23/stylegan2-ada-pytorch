import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

# Default paths relative to the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_PATH = REPO_ROOT / "data" / "dataset" / "dataset_split.json"
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "dataset" / "BTXRD_resized_sorted2"


def load_train_indices(split_path: Path) -> List[int]:
    """Load only the train indices from dataset_split.json."""
    split_data = json.loads(split_path.read_text())
    try:
        train_split = split_data["train"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{split_path} must contain a 'train' list") from exc

    if not isinstance(train_split, list):
        raise ValueError(f"'train' in {split_path} must be a list of integers.")

    try:
        return [int(v) for v in train_split]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'train' in {split_path} must be a list of integers.") from exc


def load_labels(dataset_json: Path) -> List[Tuple[str, int]]:
    data = json.loads(dataset_json.read_text())
    try:
        return data["labels"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{dataset_json} must contain a 'labels' list") from exc


def rewrite_dataset_json(dataset_json: Path, labels: Sequence[Tuple[str, int]]) -> None:
    dataset_json.write_text(json.dumps({"labels": list(labels)}))


def delete_not_in_split(
    source_dir: Path, labels: Sequence[Tuple[str, int]], train_indices: Sequence[int], dry_run: bool
) -> Tuple[int, int, List[str], List[Tuple[str, int]]]:
    max_index = len(labels) - 1
    for idx in train_indices:
        if idx < 0 or idx > max_index:
            raise IndexError(f"Index {idx} from split is outside the label range 0-{max_index}.")

    keep = set(train_indices)
    removed = 0
    missing: List[str] = []
    kept_labels: List[Tuple[str, int]] = []

    for idx, (rel_path, class_id) in enumerate(labels):
        file_path = source_dir / rel_path
        if idx in keep:
            kept_labels.append((rel_path, class_id))
            continue

        if file_path.exists():
            if not dry_run:
                file_path.unlink()
            removed += 1
        else:
            missing.append(rel_path)

    return removed, len(kept_labels), missing, kept_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete BTXRD_resized_sorted2 images whose index is NOT listed in the train split of dataset_split.json."
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
        help=f"Directory with BTXRD_resized_sorted2 (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without removing files or rewriting dataset.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_json = args.dataset_dir / "dataset.json"

    train_indices = load_train_indices(args.split_path)
    labels = load_labels(dataset_json)

    removed, kept, missing, kept_labels = delete_not_in_split(
        source_dir=args.dataset_dir, labels=labels, train_indices=train_indices, dry_run=args.dry_run
    )

    if not args.dry_run:
        rewrite_dataset_json(dataset_json, kept_labels)

    print(f"Kept {kept} train images from dataset_split.json.")
    print(f"Removed {removed} images not in the train split{' (dry run)' if args.dry_run else ''}.")
    if missing:
        print(f"{len(missing)} entries were missing on disk (not deleted):")
        for rel_path in missing:
            print(f"  {rel_path}")
    if args.dry_run:
        print("Dry run only — rerun without --dry-run to apply deletions and rewrite dataset.json.")


if __name__ == "__main__":
    main()
