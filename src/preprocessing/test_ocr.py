"""
Quick OCR smoke test.

Purpose:
1. Load one sampled document image.
2. Run Tesseract OCR without preprocessing.
3. Print the extracted text.

This gives us a raw OCR baseline before we add
deskewing, binarisation or denoising.
"""

from pathlib import Path

import pytesseract
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Change this path to any sampled image you want to test.
IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "sampled"
    / "invoice"
)


def get_first_image(folder: Path) -> Path:
    """
    Return the first PNG image found in the given folder.
    """

    images = sorted(folder.glob("*.png"))

    if not images:
        raise FileNotFoundError(
            f"No PNG images found in: {folder}"
        )

    return images[0]


def main():
    image_path = get_first_image(IMAGE_PATH)

    print(f"Using image:\n{image_path}\n")

    image = Image.open(image_path)

    text = pytesseract.image_to_string(
        image,
        lang="eng+fra",
    )

    print("=" * 60)
    print("RAW OCR OUTPUT")
    print("=" * 60)

    print(text)

    print("=" * 60)


if __name__ == "__main__":
    main()