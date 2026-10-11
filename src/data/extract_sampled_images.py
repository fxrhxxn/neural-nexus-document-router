"""
Extract the selected RVL-CDIP images.

This script:
1. Loads the source indices selected during reservoir sampling.
2. Streams the RVL-CDIP training split.
3. Saves only the selected 6,750 images.
4. Creates sampled_dataset.csv containing metadata for each image.

The extraction is resumable:
if an image already exists locally, it is not downloaded/saved again.
"""

import csv
import json
from pathlib import Path

from datasets import load_dataset

from src.config import (
    HF_DATASET_NAME,
    HF_DATASET_SPLIT,
    DOCUMENT_TO_ROUTING_OUTCOME,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

METADATA_DIR = DATA_DIR / "metadata"

SAMPLED_DATA_DIR = DATA_DIR / "sampled"

SAMPLE_INDICES_PATH = METADATA_DIR / "sample_indices.json"

SAMPLED_DATASET_CSV = METADATA_DIR / "sampled_dataset.csv"


def load_selected_indices():
    """
    Load the selected source indices from sample_indices.json.

    Returns:
        selected_lookup:
            Dictionary mapping source_index -> class information.

    Example:
        {
            120011: {
                "class_id": 6,
                "class_name": "invoice"
            }
        }
    """

    with open(
        SAMPLE_INDICES_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        sampling_data = json.load(file)

    selected_lookup = {}

    for class_name, class_info in sampling_data["classes"].items():

        class_id = class_info["class_id"]

        for source_index in class_info["selected_indices"]:

            selected_lookup[source_index] = {
                "class_id": class_id,
                "class_name": class_name,
            }

    return selected_lookup


def create_class_directories(selected_lookup):
    """
    Create one local directory for each selected document class.
    """

    class_names = {
        info["class_name"]
        for info in selected_lookup.values()
    }

    for class_name in class_names:

        class_directory = (
            SAMPLED_DATA_DIR /
            class_name
        )

        class_directory.mkdir(
            parents=True,
            exist_ok=True
        )


def extract_images(dataset, selected_lookup):
    """
    Stream RVL-CDIP and save only the records selected
    during reservoir sampling.

    Returns:
        metadata_rows:
            Metadata describing every selected local image.
    """

    metadata_rows = []

    total_required = len(selected_lookup)

    extracted_count = 0

    print(
        f"\nImages required: {total_required}"
    )

    print("Starting extraction...\n")

    # --------------------------------------------------------
    # Iterate through the original RVL-CDIP stream.
    # --------------------------------------------------------

    for source_index, sample in enumerate(dataset):

        # Ignore documents that were not selected.
        if source_index not in selected_lookup:
            continue

        selected_info = selected_lookup[source_index]

        class_id = selected_info["class_id"]

        class_name = selected_info["class_name"]

        routing_outcome = (
            DOCUMENT_TO_ROUTING_OUTCOME[class_name]
        )

        # ----------------------------------------------------
        # Create a stable filename.
        # ----------------------------------------------------

        filename = (
            f"{class_name}_{source_index}.png"
        )

        relative_image_path = (
            Path("data") /
            "sampled" /
            class_name /
            filename
        )

        absolute_image_path = (
            PROJECT_ROOT /
            relative_image_path
        )

        # ----------------------------------------------------
        # Save the image.
        #
        # If the file already exists, we skip saving it.
        # This makes the script resumable.
        # ----------------------------------------------------

        if not absolute_image_path.exists():

            image = sample["image"]

            # Convert to grayscale explicitly so all saved
            # documents use a consistent image mode.
            image = image.convert("L")

            image.save(
                absolute_image_path,
                format="PNG"
            )

        # ----------------------------------------------------
        # Record metadata.
        # ----------------------------------------------------

        metadata_rows.append(
            {
                "source_index": source_index,
                "class_id": class_id,
                "original_class": class_name,
                "routing_outcome": routing_outcome,
                "image_path": str(
                    relative_image_path
                ).replace("\\", "/"),
            }
        )

        extracted_count += 1

        # Show progress periodically.
        if (
            extracted_count % 100 == 0
            or extracted_count == total_required
        ):

            print(
                f"Extracted "
                f"{extracted_count}/{total_required}"
            )

        # ----------------------------------------------------
        # Stop once all selected images have been found.
        # ----------------------------------------------------

        if extracted_count == total_required:
            break

    return metadata_rows


def save_metadata_csv(metadata_rows):
    """
    Save metadata for the sampled dataset.
    """

    fieldnames = [
        "source_index",
        "class_id",
        "original_class",
        "routing_outcome",
        "image_path",
    ]

    with open(
        SAMPLED_DATASET_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(metadata_rows)

    print(
        f"\nMetadata saved to:\n"
        f"{SAMPLED_DATASET_CSV}"
    )


def validate_extraction(metadata_rows, selected_lookup):
    """
    Verify that all expected selected documents were extracted.
    """

    expected = len(selected_lookup)

    actual = len(metadata_rows)

    print("\nValidating extraction...")

    print(
        f"Expected documents: {expected}"
    )

    print(
        f"Extracted documents: {actual}"
    )

    if actual != expected:

        raise ValueError(
            "Extraction incomplete. "
            f"Expected {expected} documents "
            f"but extracted {actual}."
        )

    print(
        "Extraction validation successful."
    )


def main():

    print(
        "Loading selected RVL-CDIP indices..."
    )

    selected_lookup = (
        load_selected_indices()
    )

    print(
        f"Selected indices loaded: "
        f"{len(selected_lookup)}"
    )

    # Create folders before extraction begins.
    create_class_directories(
        selected_lookup
    )

    print(
        "\nLoading RVL-CDIP dataset..."
    )

    dataset = load_dataset(
        HF_DATASET_NAME,
        split=HF_DATASET_SPLIT,
        streaming=True,
    )

    print(
        "Dataset stream ready."
    )

    metadata_rows = extract_images(
        dataset,
        selected_lookup,
    )

    validate_extraction(
        metadata_rows,
        selected_lookup,
    )

    save_metadata_csv(
        metadata_rows
    )

    print(
        "\nImage extraction completed successfully."
    )


if __name__ == "__main__":
    main()