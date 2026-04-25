from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw YOLO labels on random images for manual verification.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/label_checks"))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []

    for split in ("train", "val"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split

        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                pairs.append((image_path, label_path))

    return pairs


def parse_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for raw_line in label_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        class_id, x_center, y_center, width, height = line.split()
        boxes.append((int(class_id), float(x_center), float(y_center), float(width), float(height)))
    return boxes


def yolo_to_pixels(box: tuple[int, float, float, float, float], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    _, x_center, y_center, width, height = box

    box_width = width * image_width
    box_height = height * image_height
    center_x = x_center * image_width
    center_y = y_center * image_height

    left = int(round(center_x - box_width / 2.0))
    top = int(round(center_y - box_height / 2.0))
    right = int(round(center_x + box_width / 2.0))
    bottom = int(round(center_y + box_height / 2.0))

    return left, top, right, bottom


def draw_boxes(image_path: Path, label_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box in parse_yolo_labels(label_path):
        left, top, right, bottom = yolo_to_pixels(box, image.width, image.height)
        draw.rectangle((left, top, right, bottom), outline="lime", width=3)
        text = "plate"
        text_bbox = draw.textbbox((left, max(0, top - 14)), text, font=font)
        draw.rectangle(text_bbox, fill="lime")
        draw.text((text_bbox[0], text_bbox[1]), text, fill="black", font=font)

    return image


def save_contact_sheet(images: list[Image.Image], output_path: Path, thumb_size: tuple[int, int] = (420, 280)) -> None:
    columns = 2
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb_size[0], rows * thumb_size[1]), color="white")

    for index, image in enumerate(images):
        thumb = ImageOps.fit(image, thumb_size)
        x_offset = (index % columns) * thumb_size[0]
        y_offset = (index // columns) * thumb_size[1]
        canvas.paste(thumb, (x_offset, y_offset))

    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(args.dataset_root)
    if not pairs:
        raise FileNotFoundError(
            f"No image/label pairs found under {args.dataset_root}. Run convert.py first."
        )

    sample_count = min(args.num_samples, len(pairs))
    selected_pairs = random.sample(pairs, sample_count)

    rendered_images: list[Image.Image] = []

    for index, (image_path, label_path) in enumerate(selected_pairs, start=1):
        rendered = draw_boxes(image_path, label_path)
        output_path = args.output_dir / f"{index:02d}_{image_path.name}"
        rendered.save(output_path)
        rendered_images.append(rendered)
        print(f"Saved visualization: {output_path}")

    contact_sheet_path = args.output_dir / "montage.jpg"
    save_contact_sheet(rendered_images, contact_sheet_path)
    print(f"Saved contact sheet: {contact_sheet_path}")


if __name__ == "__main__":
    main()
