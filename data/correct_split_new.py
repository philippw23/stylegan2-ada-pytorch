import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

# Default paths relative to the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_PATH = REPO_ROOT / "data" / "dataset" / "dataset_split.json"
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "dataset" / "BTXRD_resized_sorted"
DEFAULT_INDEX_MAP = REPO_ROOT / "data" / "dataset" / "final_patched_index_map.json"


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


def load_index_map(index_map_path: Path) -> dict:
    data = json.loads(index_map_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{index_map_path} must contain a JSON dictionary of index -> filename.")
    return data


def rewrite_dataset_json(dataset_json: Path, labels: Sequence[Tuple[str, int]]) -> None:
    dataset_json.write_text(json.dumps({"labels": list(labels)}))


def delete_not_in_split(
    source_dir: Path,
    labels: Sequence[Tuple[str, int]],
    train_indices: Sequence[int],
    index_map: dict,
    dry_run: bool,
) -> Tuple[int, int, List[str], List[Tuple[str, int]], List[str]]:
    missing_index = [idx for idx in train_indices if str(idx) not in index_map]
    if missing_index:
        raise KeyError(f"Index map missing {len(missing_index)} train indices (e.g. {missing_index[0]}).")

    keep_names = {index_map[str(idx)] for idx in train_indices}
    removed = 0
    missing: List[str] = []
    removed_names: List[str] = []
    kept_labels: List[Tuple[str, int]] = []

    for idx, (rel_path, class_id) in enumerate(labels):
        file_path = source_dir / rel_path
        if Path(rel_path).name in keep_names:
            kept_labels.append((rel_path, class_id))
            continue

        if file_path.exists():
            if not dry_run:
                file_path.unlink()
            removed += 1
            removed_names.append(rel_path)
        else:
            missing.append(rel_path)

    return removed, len(kept_labels), missing, kept_labels, removed_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete images not listed in the train split, using a final_patched index map to match filenames."
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
        "--index-map",
        type=Path,
        default=DEFAULT_INDEX_MAP,
        help=f"Path to final_patched index map JSON (default: {DEFAULT_INDEX_MAP})",
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
    index_map = load_index_map(args.index_map)

    removed, kept, missing, kept_labels, removed_names = delete_not_in_split(
        source_dir=args.dataset_dir,
        labels=labels,
        train_indices=train_indices,
        index_map=index_map,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        rewrite_dataset_json(dataset_json, kept_labels)

    print(f"Kept {kept} train images from dataset_split.json.")
    print(f"Removed {removed} images not in the train split{' (dry run)' if args.dry_run else ''}.")
    if removed_names:
        print("Removed files:")
        for rel_path in removed_names:
            print(f"  {rel_path}")
    if missing:
        print(f"{len(missing)} entries were missing on disk (not deleted):")
        for rel_path in missing:
            print(f"  {rel_path}")
    if args.dry_run:
        print("Dry run only — rerun without --dry-run to apply deletions and rewrite dataset.json.")


if __name__ == "__main__":
    main()
