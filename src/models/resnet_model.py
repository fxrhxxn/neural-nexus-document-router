import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_device():
    """Return the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_resnet18(num_classes=5, pretrained=True, freeze_backbone=False):
    """
    Build a ResNet-18 model for document classification.

    Args:
        num_classes: Number of output classes, e.g. 5 or 9.
        pretrained: Use ImageNet pretrained weights.
        freeze_backbone: Freeze convolutional layers if True.

    Returns:
        PyTorch ResNet-18 model.
    """

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    return model
