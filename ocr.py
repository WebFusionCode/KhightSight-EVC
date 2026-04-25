from __future__ import annotations

from statistics import fmean

from utils import OCRResult, clean_plate_text, enforce_indian_plate_format, explain_paddle_runtime_error, require_cv2, require_paddle_runtime, require_paddleocr


def build_ocr_engine(lang: str = "en", use_angle_cls: bool = False):
    require_paddle_runtime()
    PaddleOCR = require_paddleocr()
    constructor_attempts = [
        {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": use_angle_cls,
            "text_detection_model_name": "PP-OCRv5_mobile_det" if lang == "en" else None,
            "text_recognition_model_name": "en_PP-OCRv5_mobile_rec" if lang == "en" else None,
        },
        {
            "lang": lang,
            "use_angle_cls": use_angle_cls,
            "show_log": False,
        },
        {
            "lang": lang,
            "use_angle_cls": use_angle_cls,
        },
        {
            "lang": lang,
        },
    ]

    last_error: Exception | None = None
    for kwargs in constructor_attempts:
        try:
            return PaddleOCR(**kwargs)
        except (TypeError, ValueError) as error:
            message = str(error)
            if "Unknown argument" in message or "unexpected keyword argument" in message:
                last_error = error
                continue
            raise
        except RuntimeError as error:
            explain_paddle_runtime_error(error)
            raise SystemExit(1) from error

    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to initialize PaddleOCR.")


def _line_anchor_x(line) -> float:
    points = line[0]
    return min(point[0] for point in points)

def _extract_text_and_scores_from_predict_result(result) -> tuple[list[str], list[float]]:
    payload = result
    if isinstance(payload, dict) and ("rec_texts" in payload or "rec_text" in payload):
        pass
    else:
        payload = getattr(result, "json", result)

    if not isinstance(payload, dict):
        return [], []

    texts = payload.get("rec_texts")
    scores = payload.get("rec_scores")

    if texts is None and payload.get("rec_text") is not None:
        texts = [payload["rec_text"]]
    if scores is None and payload.get("rec_score") is not None:
        scores = [payload["rec_score"]]

    if texts is None:
        texts = []
    if scores is None:
        scores = []

    return [str(text) for text in texts], [float(score) for score in scores]

def run_ocr(ocr_engine, preprocessed_plate) -> OCRResult:
    cv2 = require_cv2()
    paddle_input = cv2.cvtColor(preprocessed_plate, cv2.COLOR_GRAY2BGR)

    tokens: list[str] = []
    confidences: list[float] = []

    if hasattr(ocr_engine, "predict"):
        predict_result = ocr_engine.predict(paddle_input)
        first_result = predict_result[0] if predict_result else None
        if first_result is not None:
            tokens, confidences = _extract_text_and_scores_from_predict_result(first_result)

    if not tokens and hasattr(ocr_engine, "ocr"):
        # Some versions of PaddleOCR (like v2.8+ on Python 3.13) fail if 'cls' is passed
        # explicitly to ocr() because the internal predict() method doesn't accept it.
        try:
            raw_result = ocr_engine.ocr(paddle_input, cls=False)
        except TypeError:
            raw_result = ocr_engine.ocr(paddle_input)

        first_item = raw_result[0] if raw_result else None
        if isinstance(first_item, dict):
            tokens, confidences = _extract_text_and_scores_from_predict_result(first_item)
        elif isinstance(first_item, list):
            ordered_lines = sorted(first_item, key=_line_anchor_x)
            tokens = [line[1][0] for line in ordered_lines]
            confidences = [float(line[1][1]) for line in ordered_lines]

    if not tokens:
        return OCRResult(plate_text="", confidence=0.0, raw_text="")

    raw_text = "".join(tokens)
    return OCRResult(
        plate_text=raw_text,
        confidence=float(fmean(confidences)) if confidences else 0.0,
        raw_text=raw_text,
    )
