from __future__ import annotations

from pathlib import Path

from utils import PlateDetection, require_ultralytics, resolve_device, resolve_weights


def load_detector(weights: Path | None = None):
    YOLO = require_ultralytics()
    resolved_weights = resolve_weights(weights)
    return YOLO(str(resolved_weights))


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


def detect_plates(model, image, imgsz: int = 640, conf: float = 0.25, device: str = "auto") -> list[PlateDetection]:
    resolved_device = resolve_device(device)
    result = model.predict(source=image, imgsz=imgsz, conf=conf, device=resolved_device, verbose=False)[0]

    detections: list[PlateDetection] = []
    if result.boxes is None:
        return detections

    boxes_xyxy = result.boxes.xyxy.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()

    for bbox, score in zip(boxes_xyxy, confidences, strict=True):
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        detections.append(
            PlateDetection(
                bbox=[x1, y1, x2, y2],
                detection_confidence=float(score),
            )
        )

    return detections

def extract_crops(image, detections: list[PlateDetection], padding: int = 8) -> list[tuple]:
    """Returns a list of (cropped_image, padded_bbox) for each detection."""
    crops = []
    height, width = image.shape[:2]
    for det in detections:
        padded_bbox = pad_bbox(det.bbox, padding, width, height)
        x1, y1, x2, y2 = padded_bbox
        crop = image[y1:y2, x1:x2].copy()
        crops.append((crop, padded_bbox))
    return crops
