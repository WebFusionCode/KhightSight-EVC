from __future__ import annotations

import re

def clean_plate_text(text: str) -> str:
    """Uppercase and keep only alphanumerics."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text or "")
    return cleaned.upper()

def correct_common_mistakes(text: str) -> str:
    """Map common OCR typos to their correct representations based on character position."""
    if not text:
        return text

    # Common OCR mistakes mapping
    char_to_digit = str.maketrans({"O": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"})
    digit_to_char = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"})

    # Very generic, safe mapping on whole string if we're not enforcing strict format
    # In strict format, we'll swap chars based on positional regex expectations.
    return text

def validate_and_format_indian_plate(text: str) -> tuple[str, bool]:
    """
    Enforces Indian plate format: AA00AA0000 or variations.
    Uses regex: ^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$
    """
    cleaned = clean_plate_text(text)
    if len(cleaned) < 8:
        return cleaned, False

    digit_for_char = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"})
    char_for_digit = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"})

    def make_candidate(prefix_letters: str, district_digits: str, series_letters: str, number_digits: str) -> str:
        return (
            prefix_letters.translate(char_for_digit)
            + district_digits.translate(digit_for_char)
            + series_letters.translate(char_for_digit)
            + number_digits.translate(digit_for_char)
        )

    # Attempt to fit AA00AA0000, AA00A0000 etc. format.
    # District codes are typically 2 digits but maybe 1. Series is 1 to 3 letters.
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
                return candidate, True

    return cleaned, False

def build_json_output(plate_text: str, combined_score: float, bbox: list[int]) -> dict:
    return {
        "plate_text": plate_text,
        "confidence": combined_score,
        "bbox": bbox
    }
