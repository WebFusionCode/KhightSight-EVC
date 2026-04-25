from __future__ import annotations

import cv2
import json

from detect import detect_plates, extract_crops
from preprocess import preprocess_plate
from ocr import run_ocr
from postprocess import correct_common_mistakes, validate_and_format_indian_plate, build_json_output
from utils import draw_plate_predictions, PlatePrediction


def process_frame(
    image,
    detector,
    ocr_engine,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = "auto",
    padding: int = 8,
    enforce_indian_format: bool = True,
    spatial_cache: dict = None,
) -> tuple:
    """
    Runs ANPR on a single frame (BGR numpy array).
    Allows spatial caching across video frames.
    """
    import numpy as np

    detections = detect_plates(detector, image, imgsz=imgsz, conf=conf, device=device)
    crops_info = extract_crops(image, detections, padding=padding)

    predictions: list[PlatePrediction] = []
    results_json = []
    diagnostics = []

    for idx, (crop, padded_bbox) in enumerate(crops_info):
        if crop.size == 0:
            continue

        raw_crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        preprocessed = None

        # Simple spatial cache check for video processing (if enabled)
        cache_hit = False
        if spatial_cache is not None:
            for cached_bbox, cached_data in spatial_cache.items():
                cx1, cy1 = (padded_bbox[0]+padded_bbox[2])/2, (padded_bbox[1]+padded_bbox[3])/2
                cx2, cy2 = (cached_bbox[0]+cached_bbox[2])/2, (cached_bbox[1]+cached_bbox[3])/2
                dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                
                if dist < 20: 
                    ocr_res = cached_data
                    cache_hit = True
                    break

        if not cache_hit:
            # Bypass grayscale preprocessing to preserve the critical RGB contrast for PaddleOCR
            ocr_res = run_ocr(ocr_engine, raw_crop_rgb)

        plate_text = ocr_res.plate_text
        confidence = ocr_res.confidence

        # Post Processing
        if enforce_indian_format:
            plate_text, is_valid = validate_and_format_indian_plate(plate_text)
            if not is_valid:
                plate_text = correct_common_mistakes(plate_text)
        else:
            plate_text = correct_common_mistakes(plate_text)

        combined_score = detections[idx].detection_confidence * confidence

        if spatial_cache is not None and not cache_hit:
            spatial_cache[tuple(padded_bbox)] = ocr_res

        pred = PlatePrediction(
            plate_text=plate_text,
            confidence=confidence,
            bbox=padded_bbox,
            detection_confidence=detections[idx].detection_confidence,
        )
        predictions.append(pred)

        results_json.append(build_json_output(plate_text, float(combined_score), padded_bbox))
        
        # Save diagnostics for visualization
        diag_item = {"raw": raw_crop_rgb}
        if preprocessed is not None:
            diag_item["preprocessed"] = preprocessed # grayscale
        diagnostics.append(diag_item)

    annotated_image = draw_plate_predictions(image, predictions)
    
    return annotated_image, results_json, diagnostics
