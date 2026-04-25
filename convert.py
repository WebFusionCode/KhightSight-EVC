from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert top-left JSON annotations into a YOLOv8-ready train/val dataset."
    )
    parser.add_argument("--source-images", type=Path, default=Path("dataset/images"))
    parser.add_argument("--source-labels", type=Path, default=Path("dataset/images/labels"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--image-zip", type=Path, default=None, help="Optional zip archive containing raw images.")
    parser.add_argument("--label-zip", type=Path, default=None, help="Optional zip archive containing raw JSON labels.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {destination}")
    with ZipFile(zip_path) as zip_file:
        zip_file.extractall(destination)


def collect_top_level_images(images_dir: Path) -> list[Path]:
    image_paths = [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths)


def build_label_index(labels_dir: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted(labels_dir.rglob("*.json"))}


def clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare_output_dirs(dataset_root: Path) -> dict[str, Path]:
    image_train = dataset_root / "images" / "train"
    image_val = dataset_root / "images" / "val"
    label_train = dataset_root / "labels" / "train"
    label_val = dataset_root / "labels" / "val"

    clean_directory(image_train)
    clean_directory(image_val)
    clean_directory(label_train)
    clean_directory(label_val)

    return {
        "train_images": image_train,
        "val_images": image_val,
        "train_labels": label_train,
        "val_labels": label_val,
    }


def split_items(items: list[Path], train_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(shuffled) * train_ratio)
    train_items = shuffled[:train_count]
    val_items = shuffled[train_count:]
    if not train_items or not val_items:
        raise ValueError("Train/val split produced an empty subset. Adjust the dataset size or train_ratio.")
    return train_items, val_items


def load_annotations(label_path: Path) -> list[dict]:
    data = json.loads(label_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of annotations in {label_path}")
    return data


def convert_annotation(annotation: dict, image_width: int, image_height: int) -> str:
    x = float(annotation["x"])
    y = float(annotation["y"])
    width = float(annotation["width"])
    height = float(annotation["height"])

    x_center = (x + width / 2.0) / image_width
    y_center = (y + height / 2.0) / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height

    values = [0, x_center, y_center, normalized_width, normalized_height]
    return " ".join(
        f"{value:.6f}" if isinstance(value, float) else str(value)
        for value in values
    )


def write_yolo_label_file(output_path: Path, annotations: Iterable[dict], image_width: int, image_height: int) -> int:
    lines = [convert_annotation(annotation, image_width, image_height) for annotation in annotations]
    output_path.write_text("\n".join(lines) + "\n")
    return len(lines)


def copy_split(
    image_paths: list[Path],
    label_index: dict[str, Path],
    image_output_dir: Path,
    label_output_dir: Path,
) -> int:
    annotation_count = 0

    for image_path in image_paths:
        label_path = label_index.get(image_path.stem)
        if label_path is None:
            raise FileNotFoundError(f"Missing JSON label for {image_path.name}")

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        annotations = load_annotations(label_path)
        yolo_label_path = label_output_dir / f"{image_path.stem}.txt"

        shutil.copy2(image_path, image_output_dir / image_path.name)
        annotation_count += write_yolo_label_file(yolo_label_path, annotations, image_width, image_height)

    return annotation_count


def write_data_yaml(project_root: Path) -> Path:
    data_yaml = project_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: dataset/",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names: [\"plate\"]",
                "",
            ]
        )
    )
    return data_yaml


def write_summary(dataset_root: Path, summary: dict) -> Path:
    summary_path = dataset_root / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary_path


def resolve_sources(args: argparse.Namespace) -> tuple[Path, Path]:
    source_images = args.source_images
    source_labels = args.source_labels

    if args.image_zip:
        source_images = args.dataset_root / "_raw_images"
        clean_directory(source_images)
        extract_zip(args.image_zip, source_images)

    if args.label_zip:
        source_labels = args.dataset_root / "_raw_labels"
        clean_directory(source_labels)
        extract_zip(args.label_zip, source_labels)

    return source_images, source_labels


def main() -> None:
    args = parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")

    random.seed(args.seed)

    source_images, source_labels = resolve_sources(args)
    image_paths = collect_top_level_images(source_images)
    label_index = build_label_index(source_labels)

    if not image_paths:
        raise FileNotFoundError(f"No images found in {source_images}")

    image_stems = {path.stem for path in image_paths}
    label_stems = set(label_index)
    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    if missing_labels:
        raise FileNotFoundError(f"Missing {len(missing_labels)} labels, first few: {missing_labels[:5]}")
    if missing_images:
        print(f"Warning: {len(missing_images)} labels have no matching image. They will be ignored.")

    output_dirs = prepare_output_dirs(args.dataset_root)
    train_images, val_images = split_items(image_paths, args.train_ratio, args.seed)

    train_box_count = copy_split(train_images, label_index, output_dirs["train_images"], output_dirs["train_labels"])
    val_box_count = copy_split(val_images, label_index, output_dirs["val_images"], output_dirs["val_labels"])

    data_yaml_path = write_data_yaml(Path.cwd())
    summary = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "total_images": len(image_paths),
        "train_images": len(train_images),
        "val_images": len(val_images),
        "total_boxes": train_box_count + val_box_count,
        "train_boxes": train_box_count,
        "val_boxes": val_box_count,
        "data_yaml": str(data_yaml_path),
    }
    summary_path = write_summary(args.dataset_root, summary)

    print("Dataset conversion complete.")
    print(json.dumps(summary, indent=2))
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
