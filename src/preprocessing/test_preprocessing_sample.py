"""
Test OCR preprocessing across a representative sample.

This experiment:
1. Loads the frozen dataset split.
2. Uses training documents only.
3. Selects 2 documents from 5 different classes.
4. Runs OCR on the original images.
5. Applies upscale + Otsu thresholding.
6. Runs OCR on the preprocessed images.
7. Saves the comparison results to CSV.

The purpose is to determine whether Otsu preprocessing
consistently improves OCR across different document types.
"""

from pathlib import Path

import cv2
import pandas as pd
import pytesseract
from PIL import Image


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "dataset_split.csv"
)

OUTPUT_IMAGE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "preprocessing_sample"
)

OUTPUT_CSV_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "preprocessing_sample"
    / "ocr_preprocessing_comparison.csv"
)


# ============================================================
# Experiment configuration
# ============================================================

CLASSES_TO_TEST = [
    "invoice",
    "letter",
    "form",
    "resume",
    "handwritten",
]

SAMPLES_PER_CLASS = 2

RANDOM_SEED = 42


def select_sample_documents():
    """
    Select a small reproducible sample from the training set.

    Two documents are selected from each of five document
    classes, giving 10 test documents in total.
    """

    dataframe = pd.read_csv(
        DATASET_SPLIT_PATH
    )

    # Only use training documents for this experiment.
    train_df = dataframe[
        dataframe["split"] == "train"
    ]

    selected_rows = []

    for class_name in CLASSES_TO_TEST:

        class_df = train_df[
            train_df["original_class"]
            == class_name
        ]

        sampled = class_df.sample(
            n=SAMPLES_PER_CLASS,
            random_state=RANDOM_SEED,
        )

        selected_rows.append(
            sampled
        )

    sample_df = pd.concat(
        selected_rows,
        ignore_index=True,
    )

    return sample_df


def preprocess_otsu(image_path: Path):
    """
    Apply the Otsu preprocessing approach that performed
    best in the initial single-document experiment.

    Steps:
    1. Load image as grayscale.
    2. Upscale by 1.5x.
    3. Apply Otsu thresholding.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}"
        )

    enlarged = cv2.resize(
        image,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC,
    )

    _, thresholded = cv2.threshold(
        enlarged,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    return thresholded


def run_raw_ocr(image_path: Path):
    """
    Run bilingual OCR on the original image.
    """

    image = Image.open(
        image_path
    )

    text = pytesseract.image_to_string(
        image,
        lang="eng+fra",
    )

    return text


def run_processed_ocr(processed_image):
    """
    Run bilingual OCR on an OpenCV processed image.
    """

    text = pytesseract.image_to_string(
        processed_image,
        lang="eng+fra",
    )

    return text


def run_experiment(sample_df):
    """
    Run raw and Otsu-preprocessed OCR on every selected document.
    """

    OUTPUT_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    total_documents = len(
        sample_df
    )

    print(
        f"Documents selected: {total_documents}\n"
    )

    for number, (_, row) in enumerate(
        sample_df.iterrows(),
        start=1,
    ):

        source_index = row[
            "source_index"
        ]

        class_name = row[
            "original_class"
        ]

        relative_path = Path(
            row["image_path"]
        )

        original_path = (
            PROJECT_ROOT
            / relative_path
        )

        print(
            f"[{number}/{total_documents}] "
            f"{class_name} - "
            f"source index {source_index}"
        )

        # ----------------------------------------------------
        # Raw OCR
        # ----------------------------------------------------

        raw_text = run_raw_ocr(
            original_path
        )

        # ----------------------------------------------------
        # Otsu preprocessing
        # ----------------------------------------------------

        processed_image = preprocess_otsu(
            original_path
        )

        processed_filename = (
            f"{class_name}_"
            f"{source_index}_otsu.png"
        )

        processed_path = (
            OUTPUT_IMAGE_DIR
            / processed_filename
        )

        cv2.imwrite(
            str(processed_path),
            processed_image,
        )

        # ----------------------------------------------------
        # OCR after preprocessing
        # ----------------------------------------------------

        otsu_text = run_processed_ocr(
            processed_image
        )

        # ----------------------------------------------------
        # Save experiment result
        # ----------------------------------------------------

        results.append(
            {
                "source_index": source_index,
                "original_class": class_name,
                "original_image": str(
                    relative_path
                ).replace("\\", "/"),
                "processed_image": str(
                    processed_path.relative_to(
                        PROJECT_ROOT
                    )
                ).replace("\\", "/"),
                "raw_ocr_text": raw_text,
                "otsu_ocr_text": otsu_text,
            }
        )

    return pd.DataFrame(
        results
    )


def save_results(results_df):
    """
    Save OCR comparison results for later review.
    """

    results_df.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
        encoding="utf-8",
    )

    print(
        f"\nComparison results saved to:\n"
        f"{OUTPUT_CSV_PATH}"
    )


def main():

    print(
        "Selecting representative "
        "training documents...\n"
    )

    sample_df = (
        select_sample_documents()
    )

    print(
        sample_df[
            [
                "source_index",
                "original_class",
            ]
        ]
    )

    print(
        "\nRunning OCR comparison...\n"
    )

    results_df = run_experiment(
        sample_df
    )

    save_results(
        results_df
    )

    print(
        "\nPreprocessing sample experiment complete."
    )


if __name__ == "__main__":
    main()