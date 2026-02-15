"""Preprocess BTXRD datasets for StyleGAN2-ADA.

This module provides three workflows:
- ``preprocess``: resize/sort images and write optional ``dataset.json``.
- ``build-index-map``: build a deterministic index->filename JSON map.
- ``correct-split``: keep train-split images only and rewrite ``dataset.json``.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

# Default paths relative to the repository root (override via CLI flags).
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "dataset" / "final_patched_BTXRD"
DEFAULT_JSON_DIR = REPO_ROOT / "data" / "dataset" / "BTXRD" / "Annotations"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "dataset" / "BTXRD_resized_sorted_with_anatomical_location"
DEFAULT_XLSX_PATH = REPO_ROOT / "data" / "dataset" / "BTXRD" / "dataset.xlsx"
DEFAULT_SPLIT_PATH = REPO_ROOT / "data" / "dataset" / "dataset_split.json"
DEFAULT_INDEX_MAP_OUTPUT = REPO_ROOT / "data" / "dataset" / "final_patched_index_map.json"
DEFAULT_CORRECT_SPLIT_DIR = REPO_ROOT / "data" / "dataset" / "BTXRD_resized_sorted"

TUMOR_CLASSES = [
    "osteochondroma",
    "osteosarcoma",
    "multiple osteochondromas",
    "simple bone cyst",
    "giant cell tumor",
    "synovial osteochondroma",
    "osteofibroma",
]

ANATOMICAL_LOCATIONS = ["upper limb", "lower limb", "pelvis"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def load_json(path: Path) -> dict:
    """Load and parse a JSON file from disk."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_class_from_json(js: dict) -> Optional[str]:
    """Extract class label from a BTXRD-style annotation JSON object."""
    try:
        return js["shapes"][0]["label"]
    except (KeyError, IndexError, TypeError):
        return None


def center_crop_to_square(img: Image.Image) -> Image.Image:
    """Return a centered square crop using the shorter image side."""
    width, height = img.size
    if width == height:
        return img
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))


def _normalize_col_name(name: str) -> str:
    """Normalize column names for case/format-insensitive matching."""
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def load_location_by_image_id(xlsx_path: Path, anatomical_locations: List[str]) -> Dict[str, str]:
    """Load anatomical locations keyed by image_id from an XLSX sheet."""
    import pandas as pd

    df = pd.read_excel(xlsx_path)

    normalized_to_original = {
        _normalize_col_name(col): col for col in df.columns if str(col).strip() != ""
    }

    image_id_col = normalized_to_original.get("imageid")
    if image_id_col is None:
        raise ValueError(
            f"Expected an 'image_id' column in {xlsx_path} (case/format-insensitive)"
        )

    location_by_image_id: Dict[str, str] = {}
    for _, row in df.iterrows():
        image_id_raw = row.get(image_id_col)
        if image_id_raw is None or (isinstance(image_id_raw, float) and pd.isna(image_id_raw)):
            continue
        image_id = os.path.basename(str(image_id_raw).strip()).lower()
        if not image_id:
            continue

        chosen_location: Optional[str] = None
        for loc in anatomical_locations:
            original_col = normalized_to_original.get(_normalize_col_name(loc))
            if original_col is None:
                continue
            value = row.get(original_col)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            try:
                if int(value) == 1:
                    chosen_location = loc
                    break
            except (TypeError, ValueError):
                continue

        location_by_image_id[image_id] = chosen_location or "unknown"

    return location_by_image_id


def process_dataset(
    image_dir: Path,
    json_dir: Path,
    output_dir: Path,
    target_size: int,
    crop_to_square: bool = False,
    write_dataset_json: bool = True,
    use_anatomical_location: bool = False,
    xlsx_path: Optional[Path] = None,
    anatomical_locations: Optional[List[str]] = None,
) -> None:
    """Run image preprocessing and optionally write `dataset.json`."""
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    processed = 0
    skipped_missing_json = 0
    skipped_missing_label = 0
    skipped_missing_location = 0
    skipped_unknown_class = 0
    label_entries: List[Tuple[str, str]] = []
    anatomical_locations = anatomical_locations or ANATOMICAL_LOCATIONS

    if use_anatomical_location:
        if xlsx_path is None:
            raise ValueError("xlsx_path must be provided when use_anatomical_location is enabled.")
        location_by_image_id = load_location_by_image_id(xlsx_path, anatomical_locations)
    else:
        location_by_image_id = {}

    for img_path in files:
        img_id = img_path.stem
        json_path = json_dir / f"{img_id}.json"
        if not json_path.exists():
            print(f"No JSON found for {img_path.name}")
            skipped_missing_json += 1
            continue

        js = load_json(json_path)
        class_name = get_class_from_json(js)
        if not class_name:
            print(f"Cannot extract class for {img_path.name}")
            skipped_missing_label += 1
            continue
        class_name = class_name.lower()

        if use_anatomical_location and class_name not in TUMOR_CLASSES:
            print(f"Unknown tumor class for {img_path.name}: {class_name}")
            skipped_unknown_class += 1
            continue

        if use_anatomical_location:
            image_id_key = img_path.name.lower()
            location = location_by_image_id.get(image_id_key, "unknown")
            if location == "unknown":
                print(f"No anatomical location for {img_path.name}")
                skipped_missing_location += 1
                continue
            class_name = f"{location} {class_name}"

        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(img_path).convert("RGB")
        if crop_to_square:
            img = center_crop_to_square(img)
        img = img.resize((target_size, target_size), Image.LANCZOS)

        out_path = class_dir / img_path.name
        img.save(out_path)

        processed += 1
        label_entries.append((out_path.relative_to(output_dir).as_posix(), class_name))

    print("\nDONE! Images sorted and resized.")
    print(f"Processed: {processed}")
    print(f"Missing JSON: {skipped_missing_json}")
    print(f"Missing label: {skipped_missing_label}")
    if skipped_unknown_class:
        print(f"Unknown tumor class (skipped): {skipped_unknown_class}")
    if use_anatomical_location:
        print(f"Missing anatomical location: {skipped_missing_location}")

    if write_dataset_json and label_entries:
        if use_anatomical_location:
            class_names = [f"{loc} {tumor}" for loc in anatomical_locations for tumor in TUMOR_CLASSES]
        else:
            class_names = sorted({class_name for _, class_name in label_entries})
        class_to_id: Dict[str, int] = {class_name: idx for idx, class_name in enumerate(class_names)}
        labels = [[path, class_to_id[class_name]] for path, class_name in label_entries]
        dataset_path = output_dir / "dataset.json"
        dataset_path.write_text(json.dumps({"labels": labels}), encoding="utf-8")
        print(f"\nWrote dataset.json with {len(labels)} labels.")


def load_split_data(split_path: Path) -> dict:
    """Load split JSON as dictionary."""
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(split_data, dict):
        raise ValueError(f"{split_path} must contain a JSON object.")
    return split_data


def _parse_index_list(value: object, split_name: str, split_path: Path) -> List[int]:
    """Parse one split list to integer indices."""
    if not isinstance(value, list):
        raise ValueError(f"'{split_name}' in {split_path} must be a list of integers.")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{split_name}' in {split_path} must be a list of integers.") from exc


def load_split_indices(split_path: Path) -> List[int]:
    """Load and merge split indices from JSON (train/test, optional val)."""
    split_data = load_split_data(split_path)
    try:
        train_indices = _parse_index_list(split_data["train"], "train", split_path)
        test_indices = _parse_index_list(split_data["test"], "test", split_path)
    except KeyError as exc:
        raise ValueError(f"{split_path} must contain at least 'train' and 'test' lists") from exc

    val_indices = _parse_index_list(split_data.get("val", []), "val", split_path)
    return train_indices + val_indices + test_indices


def list_images(dataset_dir: Path) -> List[str]:
    """List `.jpeg` filenames in deterministic lexicographic order."""
    return sorted([path.name for path in dataset_dir.glob("*.jpeg")])


def validate_indices(indices: Sequence[int], total_images: int) -> None:
    """Validate split indices are unique and within dataset bounds."""
    if not indices:
        raise ValueError("Split indices are empty.")

    unique_indices = set(indices)
    if len(unique_indices) != len(indices):
        raise ValueError("Split indices contain duplicates.")

    min_index = min(unique_indices)
    max_index = max(unique_indices)
    if min_index < 0 or max_index > total_images - 1:
        raise ValueError(
            f"Split indices must be within 0-{total_images - 1}, got {min_index}-{max_index}."
        )


def build_index_map(images: Sequence[str]) -> Dict[str, str]:
    """Build JSON-serializable mapping from string index to image filename."""
    return {str(idx): name for idx, name in enumerate(images)}


def run_build_index_map(split_path: Path, dataset_dir: Path, output_path: Path) -> None:
    """Validate split coverage and write the final index map JSON."""
    images = list_images(dataset_dir)
    if not images:
        raise SystemExit(f"No .jpeg files found in {dataset_dir}")

    indices = load_split_indices(split_path)
    validate_indices(indices, total_images=len(images))

    index_map = build_index_map(images)
    output_path.write_text(json.dumps(index_map, indent=2), encoding="utf-8")
    print(f"Wrote {len(index_map)} entries to {output_path}")


def load_train_indices(split_path: Path) -> List[int]:
    """Load train indices from split JSON."""
    split_data = load_split_data(split_path)
    try:
        return _parse_index_list(split_data["train"], "train", split_path)
    except KeyError as exc:
        raise ValueError(f"{split_path} must contain a 'train' list") from exc


def load_labels(dataset_json: Path) -> List[Tuple[str, int]]:
    """Load `labels` entries from a StyleGAN-style `dataset.json` file."""
    data = json.loads(dataset_json.read_text(encoding="utf-8"))
    try:
        return data["labels"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{dataset_json} must contain a 'labels' list") from exc


def load_index_map(index_map_path: Path) -> dict:
    """Load index-to-filename mapping JSON."""
    data = json.loads(index_map_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{index_map_path} must contain a JSON dictionary of index -> filename.")
    return data


def rewrite_dataset_json(dataset_json: Path, labels: Sequence[Tuple[str, int]]) -> None:
    """Rewrite `dataset.json` using a filtered labels list."""
    dataset_json.write_text(json.dumps({"labels": list(labels)}), encoding="utf-8")


def delete_not_in_split(
    source_dir: Path,
    labels: Sequence[Tuple[str, int]],
    train_indices: Sequence[int],
    index_map: dict,
    dry_run: bool,
) -> Tuple[int, int, List[str], List[Tuple[str, int]], List[str]]:
    """Delete files not in train split and return deletion/retention statistics."""
    missing_index = [idx for idx in train_indices if str(idx) not in index_map]
    if missing_index:
        raise KeyError(f"Index map missing {len(missing_index)} train indices (e.g. {missing_index[0]}).")

    keep_names = {index_map[str(idx)] for idx in train_indices}
    removed = 0
    missing: List[str] = []
    removed_names: List[str] = []
    kept_labels: List[Tuple[str, int]] = []

    for rel_path, class_id in labels:
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


def run_correct_split(split_path: Path, dataset_dir: Path, index_map_path: Path, dry_run: bool) -> None:
    """Apply train split filtering to dataset files and `dataset.json`."""
    dataset_json = dataset_dir / "dataset.json"

    train_indices = load_train_indices(split_path)
    labels = load_labels(dataset_json)
    index_map = load_index_map(index_map_path)

    removed, kept, missing, kept_labels, removed_names = delete_not_in_split(
        source_dir=dataset_dir,
        labels=labels,
        train_indices=train_indices,
        index_map=index_map,
        dry_run=dry_run,
    )

    if not dry_run:
        rewrite_dataset_json(dataset_json, kept_labels)

    print(f"Kept {kept} train images from {split_path}.")
    print(f"Removed {removed} images not in the train split{' (dry run)' if dry_run else ''}.")
    if removed_names:
        print("Removed files:")
        for rel_path in removed_names:
            print(f"  {rel_path}")
    if missing:
        print(f"{len(missing)} entries were missing on disk (not deleted):")
        for rel_path in missing:
            print(f"  {rel_path}")
    if dry_run:
        print("Dry run only: rerun without --dry-run to apply deletions and rewrite dataset.json.")


def run_full_pipeline(
    image_dir: Path,
    json_dir: Path,
    preprocess_output_dir: Path,
    target_size: int,
    center_crop: bool,
    write_dataset_json: bool,
    use_anatomical_location: bool,
    xlsx_path: Path,
    split_path: Path,
    index_map_dataset_dir: Path,
    index_map_output_path: Path,
    correct_split_dataset_dir: Optional[Path],
    dry_run: bool,
) -> None:
    """Run preprocess, index-map build, and train-split correction in sequence."""
    print("[1/3] Running preprocess...")
    process_dataset(
        image_dir=image_dir,
        json_dir=json_dir,
        output_dir=preprocess_output_dir,
        target_size=target_size,
        crop_to_square=center_crop,
        write_dataset_json=write_dataset_json,
        use_anatomical_location=use_anatomical_location,
        xlsx_path=xlsx_path,
    )

    print("[2/3] Building index map...")
    run_build_index_map(
        split_path=split_path,
        dataset_dir=index_map_dataset_dir,
        output_path=index_map_output_path,
    )

    print("[3/3] Applying split correction...")
    run_correct_split(
        split_path=split_path,
        dataset_dir=correct_split_dataset_dir or preprocess_output_dir,
        index_map_path=index_map_output_path,
        dry_run=dry_run,
    )


def add_preprocess_cli_args(
    parser: argparse.ArgumentParser,
    output_flag: str,
    output_default: Path,
    output_help: str,
) -> None:
    """Add preprocess-related CLI options to a parser."""
    parser.add_argument(
        "--image-dir",
        default=DEFAULT_IMAGE_DIR,
        type=Path,
        help=f"Directory with input images (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--json-dir",
        default=DEFAULT_JSON_DIR,
        type=Path,
        help=f"Directory with JSON annotations (default: {DEFAULT_JSON_DIR})",
    )
    parser.add_argument(
        output_flag,
        default=output_default,
        type=Path,
        help=output_help,
    )
    parser.add_argument("--target-size", type=int, default=256, help="Output resolution (square).")
    parser.add_argument(
        "--center-crop",
        action="store_true",
        help="Center-crop to square before resizing to avoid distortion.",
    )
    parser.add_argument(
        "--no-dataset-json",
        action="store_true",
        help="Do not write dataset.json for conditional StyleGAN2-ADA training.",
    )
    parser.add_argument(
        "--use-anatomical-location",
        action="store_true",
        help="Prefix class labels with anatomical location using dataset.xlsx metadata.",
    )
    parser.add_argument(
        "--xlsx-path",
        default=DEFAULT_XLSX_PATH,
        type=Path,
        help=f"Path to dataset.xlsx (default: {DEFAULT_XLSX_PATH}).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build command line parser for all preprocessing-related workflows."""
    parser = argparse.ArgumentParser(description="BTXRD preprocessing and split utility for StyleGAN2-ADA.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Resize/sort images and write dataset.json.")
    add_preprocess_cli_args(
        parser=preprocess,
        output_flag="--output-dir",
        output_default=DEFAULT_OUTPUT_DIR,
        output_help=f"Output directory for resized/sorted images (default: {DEFAULT_OUTPUT_DIR})",
    )

    index_map = subparsers.add_parser("build-index-map", help="Build final_patched index map.")
    index_map.add_argument(
        "--split-path",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help=f"Path to dataset_split.json (default: {DEFAULT_SPLIT_PATH})",
    )
    index_map.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Directory with final_patched_BTXRD images (default: {DEFAULT_IMAGE_DIR})",
    )
    index_map.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_INDEX_MAP_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_INDEX_MAP_OUTPUT})",
    )

    correct_split = subparsers.add_parser("correct-split", help="Filter dataset to train split.")
    correct_split.add_argument(
        "--split-path",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help=f"Path to dataset_split.json (default: {DEFAULT_SPLIT_PATH})",
    )
    correct_split.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_CORRECT_SPLIT_DIR,
        help=f"Directory with dataset.json and files to filter (default: {DEFAULT_CORRECT_SPLIT_DIR})",
    )
    correct_split.add_argument(
        "--index-map",
        type=Path,
        default=DEFAULT_INDEX_MAP_OUTPUT,
        help=f"Path to final_patched index map JSON (default: {DEFAULT_INDEX_MAP_OUTPUT})",
    )
    correct_split.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without removing files or rewriting dataset.json.",
    )

    full_pipeline = subparsers.add_parser(
        "full-pipeline",
        help="Run preprocess, build-index-map, and correct-split in one call.",
    )
    add_preprocess_cli_args(
        parser=full_pipeline,
        output_flag="--preprocess-output-dir",
        output_default=DEFAULT_OUTPUT_DIR,
        output_help=f"Output directory for resized/sorted images (default: {DEFAULT_OUTPUT_DIR})",
    )
    full_pipeline.add_argument(
        "--split-path",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help=f"Path to dataset split JSON (default: {DEFAULT_SPLIT_PATH})",
    )
    full_pipeline.add_argument(
        "--index-map-dataset-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Directory used for build-index-map (default: {DEFAULT_IMAGE_DIR})",
    )
    full_pipeline.add_argument(
        "--index-map-output-path",
        type=Path,
        default=DEFAULT_INDEX_MAP_OUTPUT,
        help=f"Output JSON path for index map (default: {DEFAULT_INDEX_MAP_OUTPUT})",
    )
    full_pipeline.add_argument(
        "--correct-split-dataset-dir",
        type=Path,
        default=None,
        help="Directory for correct-split step (default: --preprocess-output-dir).",
    )
    full_pipeline.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass through to correct-split (no deletions, no dataset.json rewrite).",
    )

    return parser


def main() -> None:
    """Dispatch to requested preprocessing subcommand."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        process_dataset(
            image_dir=args.image_dir,
            json_dir=args.json_dir,
            output_dir=args.output_dir,
            target_size=args.target_size,
            crop_to_square=args.center_crop,
            write_dataset_json=not args.no_dataset_json,
            use_anatomical_location=args.use_anatomical_location,
            xlsx_path=args.xlsx_path,
        )
    elif args.command == "build-index-map":
        run_build_index_map(
            split_path=args.split_path,
            dataset_dir=args.dataset_dir,
            output_path=args.output_path,
        )
    elif args.command == "correct-split":
        run_correct_split(
            split_path=args.split_path,
            dataset_dir=args.dataset_dir,
            index_map_path=args.index_map,
            dry_run=args.dry_run,
        )
    elif args.command == "full-pipeline":
        run_full_pipeline(
            image_dir=args.image_dir,
            json_dir=args.json_dir,
            preprocess_output_dir=args.preprocess_output_dir,
            target_size=args.target_size,
            center_crop=args.center_crop,
            write_dataset_json=not args.no_dataset_json,
            use_anatomical_location=args.use_anatomical_location,
            xlsx_path=args.xlsx_path,
            split_path=args.split_path,
            index_map_dataset_dir=args.index_map_dataset_dir,
            index_map_output_path=args.index_map_output_path,
            correct_split_dataset_dir=args.correct_split_dataset_dir,
            dry_run=args.dry_run,
        )
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
