"""
Document image preprocessing.

Current purpose:
1. Load one sampled document image.
2. Convert it to grayscale.
3. Reduce image noise.
4. Apply adaptive thresholding.
5. Save the preprocessed image for OCR comparison.

This is an initial preprocessing experiment.
Deskewing will be added later if needed.
"""

from pathlib import Path

import cv2


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "sampled"
    / "invoice"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "preprocessing_tests"
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


def preprocess_image(image_path: Path):
    """
    Apply basic preprocessing to a scanned document.

    Steps:
    1. Load image as grayscale.
    2. Apply median blur to reduce small noise.
    3. Apply adaptive thresholding to improve text contrast.

    Returns:
        Preprocessed image as a NumPy array.
    """

    # --------------------------------------------------------
    # Step 1: Load the image directly in grayscale.
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}"
        )

    # --------------------------------------------------------
    # Step 2: Reduce small image noise.
    #
    # Median blur works well for scanned document noise while
    # preserving edges around characters reasonably well.
    # --------------------------------------------------------

    denoised = cv2.medianBlur(
        image,
        3,
    )

    # --------------------------------------------------------
    # Step 3: Adaptive thresholding.
    #
    # This converts the document into a high-contrast black
    # and white image.
    #
    # Adaptive thresholding is useful when lighting or page
    # background intensity is not uniform.
    # --------------------------------------------------------

    thresholded = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )

    return thresholded


def save_preprocessed_image(
    image,
    original_path: Path,
):
    """
    Save the processed image to outputs/preprocessing_tests/.
    """

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_FOLDER
        / f"{original_path.stem}_preprocessed.png"
    )

    success = cv2.imwrite(
        str(output_path),
        image,
    )

    if not success:
        raise IOError(
            f"Could not save image to: {output_path}"
        )

    return output_path


def main():

    image_path = get_first_image(
        INPUT_FOLDER
    )

    print(
        f"Original image:\n{image_path}\n"
    )

    processed_image = preprocess_image(
        image_path
    )

    output_path = save_preprocessed_image(
        processed_image,
        image_path,
    )

    print(
        f"Preprocessed image saved to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()