from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.models.resnet_dataset import CLASS_NAMES


PREDICTIONS_PATH = Path(
    "outputs/deep_learning/resnet/"
    "partial_finetune_layer4_fc_aug/"
    "validation_predictions.csv"
)

OUTPUT_DIR = Path(
    "outputs/deep_learning/resnet/"
    "partial_finetune_layer4_fc_aug"
)

ROUTE_NAMES = [
    "Finance",
    "Correspondence",
    "HR",
    "Administration",
    "Manual Review",
]


def main():
    predictions = pd.read_csv(PREDICTIONS_PATH)

    # Document-class confusion matrix
    document_matrix = confusion_matrix(
        predictions["true_class"],
        predictions["predicted_class"],
        labels=CLASS_NAMES,
    )

    document_matrix_df = pd.DataFrame(
        document_matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    document_matrix_df.to_csv(
        OUTPUT_DIR / "validation_document_confusion_matrix.csv"
    )

    figure, axis = plt.subplots(figsize=(10, 9))

    display = ConfusionMatrixDisplay(
        confusion_matrix=document_matrix,
        display_labels=CLASS_NAMES,
    )

    display.plot(
        ax=axis,
        xticks_rotation=45,
        colorbar=False,
    )

    axis.set_title(
        "Fine-Tuned ResNet-18 Validation Confusion Matrix"
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR
        / "validation_document_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    # Routing confusion matrix
    routing_matrix = confusion_matrix(
        predictions["true_route"],
        predictions["predicted_route"],
        labels=ROUTE_NAMES,
    )

    routing_matrix_df = pd.DataFrame(
        routing_matrix,
        index=ROUTE_NAMES,
        columns=ROUTE_NAMES,
    )

    routing_matrix_df.to_csv(
        OUTPUT_DIR / "validation_routing_confusion_matrix.csv"
    )

    figure, axis = plt.subplots(figsize=(9, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=routing_matrix,
        display_labels=ROUTE_NAMES,
    )

    display.plot(
        ax=axis,
        xticks_rotation=45,
        colorbar=False,
    )

    axis.set_title(
        "Fine-Tuned ResNet-18 Validation Routing Confusion Matrix"
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR
        / "validation_routing_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    # All document-classification mistakes
    document_errors = predictions[
        ~predictions["document_correct"]
    ].copy()

    document_errors = document_errors.sort_values(
        "confidence",
        ascending=False,
    )

    document_errors.to_csv(
        OUTPUT_DIR / "validation_document_errors.csv",
        index=False,
    )

    # Only mistakes that also caused incorrect routing
    routing_errors = predictions[
        ~predictions["route_correct"]
    ].copy()

    routing_errors = routing_errors.sort_values(
        "confidence",
        ascending=False,
    )

    routing_errors.to_csv(
        OUTPUT_DIR / "validation_routing_errors.csv",
        index=False,
    )

    # Count common class confusions
    class_confusions = (
        document_errors.groupby(
            ["true_class", "predicted_class"]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False,
        )
    )

    class_confusions.to_csv(
        OUTPUT_DIR / "validation_class_confusions.csv",
        index=False,
    )

    # Count common routing confusions
    route_confusions = (
        routing_errors.groupby(
            ["true_route", "predicted_route"]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False,
        )
    )

    route_confusions.to_csv(
        OUTPUT_DIR / "validation_route_confusions.csv",
        index=False,
    )

    print("Validation analysis complete.")
    print(
        "Document errors:",
        len(document_errors),
    )
    print(
        "Routing errors:",
        len(routing_errors),
    )

    print("\nMost common document confusions:")
    print(class_confusions.head(10).to_string(index=False))

    print("\nMost common routing confusions:")
    print(route_confusions.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
