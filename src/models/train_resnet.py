from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from src.config import (
    DOCUMENT_TO_ROUTING_OUTCOME,
    RANDOM_SEED,
    ROUTING_OUTCOMES,
)
from src.models.resnet_dataset import CLASS_NAMES, NeuralNexusDataset
from src.models.resnet_model import build_resnet18, get_device


EXPERIMENT_NAME = "verification_run"
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 1
USE_AUGMENTATION = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def keep_frozen_batchnorm_in_eval(model):
    # Keep BatchNorm fixed in the frozen backbone
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def document_labels_to_routes(labels):
    return [
        DOCUMENT_TO_ROUTING_OUTCOME[CLASS_NAMES[label]]
        for label in labels
    ]


def calculate_metrics(labels, predictions):
    document_accuracy = accuracy_score(
        labels,
        predictions,
    )

    document_macro_f1 = f1_score(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        average="macro",
        zero_division=0,
    )

    true_routes = document_labels_to_routes(labels)
    predicted_routes = document_labels_to_routes(predictions)

    routing_accuracy = accuracy_score(
        true_routes,
        predicted_routes,
    )

    routing_macro_f1 = f1_score(
        true_routes,
        predicted_routes,
        labels=ROUTING_OUTCOMES,
        average="macro",
        zero_division=0,
    )

    return {
        "document_accuracy": document_accuracy,
        "document_macro_f1": document_macro_f1,
        "routing_accuracy": routing_accuracy,
        "routing_macro_f1": routing_macro_f1,
    }


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    # The backbone is frozen for this baseline
    keep_frozen_batchnorm_in_eval(model)

    running_loss = 0.0
    all_labels = []
    all_predictions = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        all_labels.extend(
            labels.detach().cpu().tolist()
        )

        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

    loss = running_loss / len(loader.dataset)

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )

    return loss, metrics


def validate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_source_indices = []

    # Validation only checks performance; it does not update the model
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            predictions = probabilities.argmax(
                dim=1
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().tolist()
            )

            all_source_indices.extend(
                batch["source_index"].tolist()
            )

    loss = running_loss / len(loader.dataset)

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )

    return (
        loss,
        metrics,
        all_labels,
        all_predictions,
        all_probabilities,
        all_source_indices,
    )


def save_predictions(
    output_dir,
    labels,
    predictions,
    probabilities,
    source_indices,
):
    rows = []

    for (
        source_index,
        true_label,
        predicted_label,
        probability_vector,
    ) in zip(
        source_indices,
        labels,
        predictions,
        probabilities,
    ):
        true_class = CLASS_NAMES[true_label]
        predicted_class = CLASS_NAMES[predicted_label]

        true_route = DOCUMENT_TO_ROUTING_OUTCOME[
            true_class
        ]

        predicted_route = DOCUMENT_TO_ROUTING_OUTCOME[
            predicted_class
        ]

        row = {
            "source_index": source_index,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "document_correct":
                true_class == predicted_class,
            "true_route": true_route,
            "predicted_route": predicted_route,
            "route_correct":
                true_route == predicted_route,
            "confidence": max(probability_vector),
        }

        for class_name, probability in zip(
            CLASS_NAMES,
            probability_vector,
        ):
            row[f"prob_{class_name}"] = probability

        rows.append(row)

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        output_dir / "validation_predictions.csv",
        index=False,
    )

    probability_columns = [
        "source_index",
        *[
            f"prob_{class_name}"
            for class_name in CLASS_NAMES
        ],
    ]

    dataframe[
        probability_columns
    ].to_csv(
        output_dir / "validation_probabilities.csv",
        index=False,
    )


def main():
    set_seed(RANDOM_SEED)

    device = get_device()

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Using device: {device}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Augmentation: {USE_AUGMENTATION}")

    train_dataset = NeuralNexusDataset(
        "train",
        use_augmentation=USE_AUGMENTATION,
    )

    validation_dataset = NeuralNexusDataset(
        "validation",
        use_augmentation=False,
    )

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"Training documents: {len(train_dataset)}"
    )

    print(
        f"Validation documents: "
        f"{len(validation_dataset)}"
    )

    print(f"Document classes: {CLASS_NAMES}")
    print(f"Routing outcomes: {ROUTING_OUTCOMES}")

    model = build_resnet18(
        num_classes=len(CLASS_NAMES),
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(
            lambda parameter:
                parameter.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
    )

    output_dir = (
        Path("outputs/deep_learning/resnet")
        / EXPERIMENT_NAME
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    best_routing_macro_f1 = -1.0
    best_epoch = None
    best_validation_data = None

    start_time = time.time()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        (
            validation_loss,
            validation_metrics,
            validation_labels,
            validation_predictions,
            validation_probabilities,
            validation_source_indices,
        ) = validate(
            model,
            validation_loader,
            criterion,
            device,
        )

        print(
            f"Train loss: {train_loss:.4f} | "
            f"9-class F1: "
            f"{train_metrics['document_macro_f1']:.4f} | "
            f"route F1: "
            f"{train_metrics['routing_macro_f1']:.4f}"
        )

        print(
            f"Validation loss: "
            f"{validation_loss:.4f} | "
            f"9-class accuracy: "
            f"{validation_metrics['document_accuracy']:.4f} | "
            f"9-class F1: "
            f"{validation_metrics['document_macro_f1']:.4f} | "
            f"route accuracy: "
            f"{validation_metrics['routing_accuracy']:.4f} | "
            f"route F1: "
            f"{validation_metrics['routing_macro_f1']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **{
                    f"train_{key}": value
                    for key, value
                    in train_metrics.items()
                },
                "validation_loss":
                    validation_loss,
                **{
                    f"validation_{key}": value
                    for key, value
                    in validation_metrics.items()
                },
            }
        )

        current_score = validation_metrics[
            "routing_macro_f1"
        ]

        # Keep the checkpoint with the best validation routing score
        if current_score > best_routing_macro_f1:
            best_routing_macro_f1 = current_score
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                output_dir / "best_resnet18.pt",
            )

            best_validation_data = (
                validation_labels,
                validation_predictions,
                validation_probabilities,
                validation_source_indices,
                validation_metrics,
            )

            print(
                "Saved new best checkpoint "
                "(routing macro-F1="
                f"{current_score:.4f})"
            )

    elapsed_seconds = time.time() - start_time

    (
        best_labels,
        best_predictions,
        best_probabilities,
        best_source_indices,
        best_metrics,
    ) = best_validation_data

    save_predictions(
        output_dir,
        best_labels,
        best_predictions,
        best_probabilities,
        best_source_indices,
    )

    document_report = classification_report(
        best_labels,
        best_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    true_routes = document_labels_to_routes(
        best_labels
    )

    predicted_routes = document_labels_to_routes(
        best_predictions
    )

    routing_report = classification_report(
        true_routes,
        predicted_routes,
        labels=ROUTING_OUTCOMES,
        output_dict=True,
        zero_division=0,
    )

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "best_epoch": best_epoch,
                **best_metrics,
            },
            file,
            indent=2,
        )

    with open(
        output_dir
        / "document_classification_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            document_report,
            file,
            indent=2,
        )

    with open(
        output_dir
        / "routing_classification_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            routing_report,
            file,
            indent=2,
        )

    with open(
        output_dir / "training_history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=2,
        )

    run_config = {
        "experiment": EXPERIMENT_NAME,
        "model": "ResNet-18",
        "pretrained": True,
        "freeze_backbone": True,
        "use_augmentation": USE_AUGMENTATION,
        "random_seed": RANDOM_SEED,
        "num_document_classes":
            len(CLASS_NAMES),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "optimizer": "Adam",
        "checkpoint_metric":
            "validation_routing_macro_f1",
        "device": str(device),
        "training_documents":
            len(train_dataset),
        "validation_documents":
            len(validation_dataset),
        "best_epoch": best_epoch,
        "best_validation_routing_macro_f1":
            best_routing_macro_f1,
        "elapsed_seconds":
            elapsed_seconds,
    }

    with open(
        output_dir / "run_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_config,
            file,
            indent=2,
        )

    print("\nExperiment complete.")
    print(f"Best epoch: {best_epoch}")

    print(
        "Best validation routing macro-F1: "
        f"{best_routing_macro_f1:.4f}"
    )

    print(
        f"Training time: "
        f"{elapsed_seconds:.1f} seconds"
    )

    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
