"""
Reproducible sampling of selected RVL-CDIP document classes.

This stage performs label-only reservoir sampling.

Goals:
1. Stream the RVL-CDIP training split.
2. Consider only the 9 document classes selected for the project.
3. Randomly select exactly 750 source records from each class.
4. Use the shared fixed random seed for reproducibility.
5. Save the selected source indices for the image extraction stage.

Images are NOT saved during this stage.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

from src.config import (
    HF_DATASET_NAME,
    HF_DATASET_SPLIT,
    RANDOM_SEED,
    SAMPLES_PER_CLASS,
    SELECTED_CLASS_IDS,
)


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

# sample_rvl_cdip.py is located at:
# project_root/src/data/sample_rvl_cdip.py
#
# parents[2] therefore points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

SAMPLE_INDICES_PATH = METADATA_DIR / "sample_indices.json"


def initialise_reservoirs():
    """
    Create an empty reservoir and counter for every selected class.

    Example:
    reservoirs[6] will eventually contain 750 source indices
    belonging to the 'invoice' class.
    """

    reservoirs = {
        class_id: []
        for class_id in SELECTED_CLASS_IDS
    }

    seen_counts = {
        class_id: 0
        for class_id in SELECTED_CLASS_IDS
    }

    return reservoirs, seen_counts


def sample_indices(dataset):
    """
    Perform reservoir sampling over the streamed dataset.

    Reservoir sampling lets us select a fixed number of records
    without loading the entire dataset into memory.

    Each selected document in the stream has a fair chance of
    appearing in the final sample.
    """

    # Fixed seed ensures that running this script again produces
    # the same random sample, assuming the dataset order is unchanged.
    rng = random.Random(RANDOM_SEED)

    reservoirs, seen_counts = initialise_reservoirs()

    print("\nSampling selected classes...\n")

    for source_index, sample in enumerate(dataset):

        label_id = sample["label"]

        # Ignore classes that are not part of our project.
        if label_id not in SELECTED_CLASS_IDS:
            continue

        # Count how many documents of this class we have encountered.
        seen_counts[label_id] += 1

        seen_for_class = seen_counts[label_id]
        reservoir = reservoirs[label_id]

        # --------------------------------------------------------
        # Fill the reservoir initially.
        # --------------------------------------------------------
        if len(reservoir) < SAMPLES_PER_CLASS:
            reservoir.append(source_index)

        # --------------------------------------------------------
        # Once the reservoir contains 750 items, later records
        # have a chance of replacing an existing selected record.
        # --------------------------------------------------------
        else:
            replacement_index = rng.randrange(seen_for_class)

            if replacement_index < SAMPLES_PER_CLASS:
                reservoir[replacement_index] = source_index

    return reservoirs, seen_counts


def validate_sampling(reservoirs):
    """
    Ensure that every selected class contains exactly
    the required number of sampled documents.
    """

    print("\nValidating sample counts...\n")

    for class_id, class_name in SELECTED_CLASS_IDS.items():

        sample_count = len(reservoirs[class_id])

        print(
            f"{class_name:<15} "
            f"(ID {class_id:>2}): "
            f"{sample_count} selected"
        )

        if sample_count != SAMPLES_PER_CLASS:
            raise ValueError(
                f"Expected {SAMPLES_PER_CLASS} samples for "
                f"'{class_name}', but found {sample_count}."
            )


def save_sample_indices(reservoirs, seen_counts):
    """
    Save the selected source indices and sampling information
    as JSON metadata.
    """

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "dataset": HF_DATASET_NAME,
        "source_split": HF_DATASET_SPLIT,
        "random_seed": RANDOM_SEED,
        "samples_per_class": SAMPLES_PER_CLASS,
        "classes": {},
    }

    for class_id, class_name in SELECTED_CLASS_IDS.items():

        output["classes"][class_name] = {
            "class_id": class_id,
            "documents_seen": seen_counts[class_id],
            "selected_indices": sorted(reservoirs[class_id]),
        }

    with open(SAMPLE_INDICES_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print(
        f"\nSampling metadata saved to:\n"
        f"{SAMPLE_INDICES_PATH}"
    )


def main():

    print("Loading RVL-CDIP dataset...")

    dataset = load_dataset(
        HF_DATASET_NAME,
        split=HF_DATASET_SPLIT,
        streaming=True,
    )

    # We only need the labels during this sampling stage.
    # Removing the image column avoids decoding every scanned page.
    dataset = dataset.remove_columns(["image"])

    print("Dataset stream ready.")

    reservoirs, seen_counts = sample_indices(dataset)

    validate_sampling(reservoirs)

    save_sample_indices(
        reservoirs,
        seen_counts,
    )

    total_selected = sum(
        len(indices)
        for indices in reservoirs.values()
    )

    print("\nSampling completed successfully.")
    print(f"Total selected documents: {total_selected}")


if __name__ == "__main__":
    main()



    """ 
    Reproducible sampling of selected RVL-CDIP document classes.
    What is it doing?
    For the first time, it streams the RVL-CDIP dataset and performs reservoir sampling to select a fixed number of documents from each of the selected classes. 
    It ensures that the sampling is reproducible by using a fixed random seed. The selected indices are then saved as JSON metadata for later use in the image extraction stage.
    ```
    invoice 1   → keep
    invoice 2   → keep
    ...
    invoice 750 → keep
    ```

    At that point, the reservoir is full. 

    This means we aren't unfairly selecting only:
    the first 750 invoices stored in the dataset.

    The final 750 are spread randomly across all invoices encountered.
    The same procedure happens independently for:
    budget
    email
    form
    handwritten
    invoice
    letter
    memo
    questionnaire
    resume


    Why RANDOM SEED 42?
    we have rng = random.Random(RANDOM_SEED)
    This ensures that running this script again produces the same random sample, assuming the dataset order is unchanged. 
    This is important for reproducibility in experiments and comparisons.
    """