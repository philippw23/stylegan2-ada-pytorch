import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

# Default paths relative to the repository root (override via CLI flags).
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "dataset" / "final_patched_BTXRD"
DEFAULT_JSON_DIR = REPO_ROOT / "data" / "dataset" / "BTXRD" / "Annotations"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "dataset" / "BTXRD_resized_sorted_with_anatomical_location"
DEFAULT_XLSX_PATH = REPO_ROOT / "data" / "dataset" / "BTXRD" / "dataset.xlsx"

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


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def get_class_from_json(js: dict) -> Optional[str]:
    """
    Extract the class label from a BTXRD-style JSON annotation.
    Adjust this function if your JSON schema differs.
    """
    try:
        return js["shapes"][0]["label"]
    except (KeyError, IndexError, TypeError):
        return None


def center_crop_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _normalize_col_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def load_location_by_image_id(
    xlsx_path: Path, anatomical_locations: List[str]
) -> Dict[str, str]:
    """
    Load anatomical locations keyed by image_id from an .xlsx file (one-hot columns).
    """
    import pandas as pd

    df = pd.read_excel(xlsx_path)

    normalized_to_original = {
        _normalize_col_name(c): c for c in df.columns if str(c).strip() != ""
    }

    image_id_col = normalized_to_original.get("imageid")
    if image_id_col is None:
        raise ValueError(
            f"Expected an 'image_id' column in {xlsx_path} (case/format-insensitive)"
        )

    location_by_image_id: Dict[str, str] = {}
    for _, row in df.iterrows():
        image_id_raw = row.get(image_id_col)
        if image_id_raw is None or (
            isinstance(image_id_raw, float) and pd.isna(image_id_raw)
        ):
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
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    processed = 0
    skipped_missing_json = 0
    skipped_missing_label = 0
    skipped_missing_location = 0
    skipped_unknown_class = 0
    label_entries: List[Tuple[str, str]] = []
    anatomical_locations = anatomical_locations or ANATOMICAL_LOCATIONS

    if use_anatomical_location:
        if xlsx_path is None:
            raise ValueError(
                "xlsx_path must be provided when use_anatomical_location is enabled."
            )
        location_by_image_id = load_location_by_image_id(xlsx_path, anatomical_locations)
    else:
        location_by_image_id = {}

    for img_path in files:
        img_id = img_path.stem
        json_path = json_dir / f"{img_id}.json"
        if not json_path.exists():
            print(f"⚠️  No JSON found for {img_path.name}")
            skipped_missing_json += 1
            continue

        js = load_json(json_path)
        class_name = get_class_from_json(js)
        if not class_name:
            print(f"⚠️  Cannot extract class for {img_path.name}")
            skipped_missing_label += 1
            continue
        class_name = class_name.lower()

        if use_anatomical_location and class_name not in TUMOR_CLASSES:
            print(f"⚠️  Unknown tumor class for {img_path.name}: {class_name}")
            skipped_unknown_class += 1
            continue

        if use_anatomical_location:
            image_id_key = img_path.name.lower()
            location = location_by_image_id.get(image_id_key, "unknown")
            if location == "unknown":
                print(f"⚠️  No anatomical location for {img_path.name}")
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

        print(f"✔️  {img_path.name} → {class_name}")
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
            class_names = [
                f"{loc} {tumor}"
                for loc in anatomical_locations
                for tumor in TUMOR_CLASSES
            ]
        else:
            class_names = sorted({c for _, c in label_entries})
        class_to_id: Dict[str, int] = {c: i for i, c in enumerate(class_names)}
        labels = [[path, class_to_id[c]] for path, c in label_entries]
        dataset_path = output_dir / "dataset.json"
        dataset_path.write_text(json.dumps({"labels": labels}))
        print(f"\nWrote dataset.json with {len(labels)} labels.")
        print("Class to id mapping:")
        for name, idx in class_to_id.items():
            print(f"  {idx}: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess BTXRD images for StyleGAN2-ADA.")
    parser.add_argument(
        "--image-dir",
        required=False,
        default=DEFAULT_IMAGE_DIR,
        type=Path,
        help=f"Directory with input images (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--json-dir",
        required=False,
        default=DEFAULT_JSON_DIR,
        type=Path,
        help=f"Directory with JSON annotations (default: {DEFAULT_JSON_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help=f"Output directory for resized/sorted images (default: {DEFAULT_OUTPUT_DIR})",
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
        help="Prefix class labels with anatomical location using dataset.xlsx metadata (3x7=21 classes).",
    )
    parser.add_argument(
        "--xlsx-path",
        required=False,
        default=DEFAULT_XLSX_PATH,
        type=Path,
        help=f"Path to dataset.xlsx containing anatomical location columns (default: {DEFAULT_XLSX_PATH}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
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
