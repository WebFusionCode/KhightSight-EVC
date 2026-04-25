from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


YOLO_CONFIG_DIR = (Path.cwd() / ".ultralytics").resolve()
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class PlateDetection:
    bbox: list[int]
    detection_confidence: float


@dataclass
class OCRResult:
    plate_text: str
    confidence: float
    raw_text: str


@dataclass
class PlatePrediction:
    plate_text: str
    confidence: float
    bbox: list[int]
    detection_confidence: float


def require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as error:
        if error.name == "cv2":
            print(
                "Missing dependency: opencv-python\n"
                "Install it with:\n"
                "  python3 -m pip install opencv-python",
                file=sys.stderr,
            )
            raise SystemExit(1) from error
        raise
    return cv2


def require_ultralytics():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        if error.name == "ultralytics":
            print(
                "Missing dependency: ultralytics\n"
                "Install it with:\n"
                "  python3 -m pip install ultralytics",
                file=sys.stderr,
            )
            raise SystemExit(1) from error
        raise
    return YOLO


def require_paddleocr():
    try:
        from paddleocr import PaddleOCR
    except ModuleNotFoundError as error:
        if error.name == "paddleocr":
            print(
                "Missing dependency: paddleocr\n"
                "Install it with:\n"
                "  python3 -m pip install paddleocr\n"
                "You also need PaddlePaddle installed for your platform.",
                file=sys.stderr,
            )
            raise SystemExit(1) from error
        raise
    return PaddleOCR


def require_paddle_runtime():
    try:
        import paddle
    except ModuleNotFoundError as error:
        if error.name == "paddle":
            print(
                "Missing dependency: paddlepaddle\n"
                "Install it with the same interpreter that runs pipeline.py.\n"
                "Example:\n"
                "  python3 -m pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/\n"
                "Then verify with:\n"
                "  python3 -c \"import paddle; print(paddle.__version__)\"",
                file=sys.stderr,
            )
            raise SystemExit(1) from error
        raise
    return paddle


def explain_paddle_runtime_error(error: Exception) -> None:
    message = str(error)
    if "dependency 'paddlepaddle' is not installed" in message or "Engine 'paddle_static' is unavailable" in message:
        print(
            "PaddleOCR is installed, but PaddlePaddle is missing in the same Python environment.\n"
            "Install PaddlePaddle with the exact interpreter you use to run pipeline.py.\n"
            "Example:\n"
            "  python3 -m pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/\n"
            "Then verify with:\n"
            "  python3 -c \"import paddle; print(paddle.__version__)\"",
            file=sys.stderr,
        )


def resolve_weights(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Weights not found: {explicit_path}")
        return explicit_path.resolve()

    # Check for root standalone deployment model
    default_model = Path("knightsight.pt")
    if default_model.exists():
        return default_model.resolve()

    candidates = sorted(Path("runs/license_plate").rglob("best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "No best.pt file found under runs/license_plate and no knightsight.pt found at root. Train the model first."
        )
    return candidates[0].resolve()


def resolve_device(device: str) -> str | None:
    if device.lower() != "auto":
        return device

    try:
        import torch
    except ModuleNotFoundError:
        return None

    mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if mps_available:
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def collect_image_paths(source: Path, limit: int | None = None) -> list[Path]:
    if source.is_file():
        return [source.resolve()]

    image_paths = sorted(
        path.resolve()
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {source}")
    if limit is not None:
        return image_paths[:limit]
    return image_paths


def load_image_bgr(image_path: Path):
    cv2 = require_cv2()
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return image


def clip_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> list[int]:
    clipped = [
        max(0, min(x1, width - 1)),
        max(0, min(y1, height - 1)),
        max(0, min(x2, width - 1)),
        max(0, min(y2, height - 1)),
    ]
    if clipped[2] <= clipped[0]:
        clipped[2] = min(width - 1, clipped[0] + 1)
    if clipped[3] <= clipped[1]:
        clipped[3] = min(height - 1, clipped[1] + 1)
    return clipped


def pad_bbox(bbox: list[int], padding: int, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    return clip_bbox(x1 - padding, y1 - padding, x2 + padding, y2 + padding, width, height)


def crop_with_padding(image, bbox: list[int], padding: int):
    height, width = image.shape[:2]
    padded_bbox = pad_bbox(bbox, padding, width, height)
    x1, y1, x2, y2 = padded_bbox
    crop = image[y1:y2, x1:x2].copy()
    return crop, padded_bbox


def clean_plate_text(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text or "")
    return cleaned.upper()


def enforce_indian_plate_format(text: str) -> str:
    cleaned = clean_plate_text(text)
    if len(cleaned) < 8:
        return cleaned

    digit_for_char = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"})
    char_for_digit = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"})

    def make_candidate(prefix_letters: str, district_digits: str, series_letters: str, number_digits: str) -> str:
        return (
            prefix_letters.translate(char_for_digit)
            + district_digits.translate(digit_for_char)
            + series_letters.translate(char_for_digit)
            + number_digits.translate(digit_for_char)
        )

    for district_len in (2, 1):
        for series_len in (2, 1, 3):
            if 2 + district_len + series_len + 4 != len(cleaned):
                continue
            candidate = make_candidate(
                cleaned[:2],
                cleaned[2 : 2 + district_len],
                cleaned[2 + district_len : 2 + district_len + series_len],
                cleaned[-4:],
            )
            pattern = rf"^[A-Z]{{2}}\d{{{district_len}}}[A-Z]{{{series_len}}}\d{{4}}$"
            if re.fullmatch(pattern, candidate):
                return candidate

    return cleaned


def prepare_text_label(plate_text: str, ocr_confidence: float, detection_confidence: float) -> str:
    text = plate_text if plate_text else "UNKNOWN"
    return f"{text} | OCR {ocr_confidence:.2f} | DET {detection_confidence:.2f}"


def draw_plate_predictions(image, predictions: list[PlatePrediction]):
    cv2 = require_cv2()
    annotated = image.copy()

    for prediction in predictions:
        x1, y1, x2, y2 = prediction.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = prepare_text_label(
            prediction.plate_text,
            prediction.confidence,
            prediction.detection_confidence,
        )
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_x = x1
        text_y = max(text_height + 8, y1 - 8)
        cv2.rectangle(
            annotated,
            (text_x, text_y - text_height - baseline - 4),
            (text_x + text_width + 4, text_y + 4),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (text_x + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    directories = {
        "annotated": output_dir / "annotated",
        "json": output_dir / "json",
        "crops": output_dir / "crops",
        "preprocessed": output_dir / "preprocessed",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def save_image(path: Path, image) -> Path:
    cv2 = require_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise ValueError(f"Failed to save image: {path}")
    return path


def save_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def predictions_to_json(predictions: list[PlatePrediction]) -> list[dict]:
    return [
        {
            "plate_text": prediction.plate_text,
            "confidence": prediction.confidence,
            "bbox": prediction.bbox,
        }
        for prediction in predictions
    ]
