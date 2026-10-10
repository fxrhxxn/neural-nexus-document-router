"""
Compare OCR results across multiple preprocessing variants.

Variants:
1. Raw image
2. Adaptive thresholding
3. Otsu thresholding

This helps us decide which preprocessing approach, if any,
actually improves OCR quality.
"""

from pathlib import Path

import pytesseract
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "sampled"
    / "invoice"
    / "invoice_120011.png"
)

ADAPTIVE_IMAGE = (
    PROJECT_ROOT
    / "outputs"
    / "preprocessing_tests"
    / "invoice_120011_preprocessed.png"
)

OTSU_IMAGE = (
    PROJECT_ROOT
    / "outputs"
    / "preprocessing_tests"
    / "invoice_120011_otsu.png"
)


def run_ocr(image_path: Path) -> str:
    """
    Run bilingual Tesseract OCR on one image.
    """

    image = Image.open(image_path)

    return pytesseract.image_to_string(
        image,
        lang="eng+fra",
    )


def print_result(title: str, image_path: Path):
    """
    Print OCR output for one preprocessing variant.
    """

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    text = run_ocr(image_path)

    print(text)


def main():

    print_result(
        "1. RAW IMAGE OCR",
        RAW_IMAGE,
    )

    print_result(
        "2. ADAPTIVE THRESHOLD OCR",
        ADAPTIVE_IMAGE,
    )

    print_result(
        "3. OTSU THRESHOLD OCR",
        OTSU_IMAGE,
    )

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()