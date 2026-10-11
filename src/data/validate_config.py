"""
Validate the shared dataset configuration.

The following checks that the selected RVL-CDIP classes and routing
mappings are internally consistent before dataset sampling begins.
"""

from src.config import (
    RVL_CDIP_LABELS,
    SELECTED_CLASS_IDS,
    DOCUMENT_TO_ROUTING_OUTCOME,
    SAMPLES_PER_CLASS,
    TOTAL_SAMPLES,
)


def main():
    print("Validating Neural Nexus dataset configuration...\n")

    # --------------------------------------------------------
    # Check selected class IDs
    # --------------------------------------------------------

    for class_id, class_name in SELECTED_CLASS_IDS.items():

        actual_name = RVL_CDIP_LABELS[class_id]

        assert actual_name == class_name, (
            f"Class mismatch for ID {class_id}: "
            f"expected '{class_name}', found '{actual_name}'"
        )

    print("✓ Selected class IDs match RVL-CDIP labels.")

    # --------------------------------------------------------
    # Check routing mappings
    # --------------------------------------------------------

    for class_name in SELECTED_CLASS_IDS.values():

        assert class_name in DOCUMENT_TO_ROUTING_OUTCOME, (
            f"No routing outcome defined for '{class_name}'."
        )

    print("✓ Every selected class has a routing outcome.")

    # --------------------------------------------------------
    # Display expected dataset size
    # --------------------------------------------------------

    print(f"\nSelected classes: {len(SELECTED_CLASS_IDS)}")
    print(f"Samples per class: {SAMPLES_PER_CLASS}")
    print(f"Expected total samples: {TOTAL_SAMPLES}")

    print("\nConfiguration validation successful.")


if __name__ == "__main__":
    main()