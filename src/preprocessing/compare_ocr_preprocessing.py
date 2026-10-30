"""
Compare OCR output before and after preprocessing.

This script:
1. Loads one original sampled invoice.
2. Loads the corresponding preprocessed image.
3. Runs Tesseract OCR on both.
4. Prints both outputs for comparison.
"""

from pathlib import Path

import pytesseract
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "sampled"
    / "invoice"
    / "invoice_120011.png"
)

PREPROCESSED_IMAGE = (
    PROJECT_ROOT
    / "outputs"
    / "preprocessing_tests"
    / "invoice_120011_preprocessed.png"
)


def run_ocr(image_path: Path) -> str:
    """
    Run bilingual Tesseract OCR on the given image.
    """

    image = Image.open(image_path)

    text = pytesseract.image_to_string(
        image,
        lang="eng+fra",
    )

    return text


def main():

    print("=" * 60)
    print("RAW IMAGE OCR")
    print("=" * 60)

    raw_text = run_ocr(
        ORIGINAL_IMAGE
    )

    print(raw_text)

    print("\n" + "=" * 60)
    print("PREPROCESSED IMAGE OCR")
    print("=" * 60)

    processed_text = run_ocr(
        PREPROCESSED_IMAGE
    )

    print(processed_text)

    print("=" * 60)


if __name__ == "__main__":
    main()