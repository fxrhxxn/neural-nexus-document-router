"""
Second preprocessing experiment.

This version uses:
1. Grayscale input.
2. Image upscaling.
3. Otsu global thresholding.

The aim is to preserve character edges better than the
first adaptive-thresholding experiment.
"""

from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "sampled"
    / "invoice"
    / "invoice_120011.png"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "preprocessing_tests"
)


def preprocess_image(image_path: Path):
    """
    Apply a gentler preprocessing pipeline.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}"
        )

    # --------------------------------------------------------
    # Step 1: Upscale image.
    #
    # This can help Tesseract recognise smaller characters.
    # --------------------------------------------------------

    enlarged = cv2.resize(
        image,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC,
    )

    # --------------------------------------------------------
    # Step 2: Otsu thresholding.
    #
    # Otsu automatically chooses one global threshold value
    # based on the image intensity distribution.
    # --------------------------------------------------------

    _, thresholded = cv2.threshold(
        enlarged,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    return thresholded


def main():

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed = preprocess_image(
        INPUT_IMAGE
    )

    output_path = (
        OUTPUT_FOLDER
        / "invoice_120011_otsu.png"
    )

    success = cv2.imwrite(
        str(output_path),
        processed,
    )

    if not success:
        raise IOError(
            f"Could not save: {output_path}"
        )

    print(
        f"Otsu-preprocessed image saved to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()