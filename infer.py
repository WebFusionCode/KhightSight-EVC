from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image

YOLO_CONFIG_DIR = (Path.cwd() / ".ultralytics").resolve()
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
try:
    from ultralytics import YOLO
except ModuleNotFoundError as error:
    if error.name == "ultralytics":
        print(
            "Missing dependency: ultralytics\n"
            "Install it with:\n"
            "  python3 -m pip install ultralytics\n"
            "If you use conda/miniforge, make sure the same python3 interpreter runs this script.",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    raise


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained YOLOv8 model.")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=Path("dataset/images/val"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/inference"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def resolve_weights(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Weights not found: {explicit_path}")
        return explicit_path

    candidates = sorted(Path("runs").rglob("best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No best.pt file found under runs/. Train the model first or pass --weights.")
    return candidates[0]


def collect_sources(source: Path, limit: int) -> list[Path]:
    if source.is_file():
        return [source]

    image_paths = sorted(
        path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {source}")
    return image_paths[:limit]


def as_ultralytics_device(device: str) -> str | None:
    if device.lower() == "auto":
        return None
    return device


def save_annotated_result(image_path: Path, annotated_array, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / image_path.name
    annotated_image = Image.fromarray(annotated_array[..., ::-1])
    annotated_image.save(save_path)
    return save_path


def main() -> None:
    args = parse_args()
    weights_path = resolve_weights(args.weights)
    source_paths = collect_sources(args.source, args.limit)
    device = as_ultralytics_device(args.device)
    output_dir = args.output_dir.resolve()

    print(f"Using weights: {weights_path}")
    print(f"Running inference on {len(source_paths)} image(s).")

    model = YOLO(str(weights_path))

    for image_path in source_paths:
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
            verbose=False,
        )[0]
        annotated = result.plot(conf=True, labels=True, line_width=2)
        save_path = save_annotated_result(image_path, annotated, output_dir)
        print(f"Saved inference result: {save_path}")


if __name__ == "__main__":
    main()
