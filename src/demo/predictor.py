"""Model hook for the demo (Uddish).

Uses Parisha's frozen ResNet-18 if the checkpoint models/best_resnet18.pt
exists. Otherwise it falls back to a clearly labelled placeholder.
Restart Streamlit after adding the checkpoint.
"""
import hashlib
from pathlib import Path

import numpy as np

from config import DOCUMENT_TO_DEPARTMENT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_resnet18.pt"

# Must match the class order used in training (src/models/resnet_dataset.py).
CLASSES = ["budget", "email", "form", "handwritten", "invoice",
           "letter", "memo", "questionnaire", "resume"]
assert set(CLASSES) == set(DOCUMENT_TO_DEPARTMENT)

USING_PLACEHOLDER = not CHECKPOINT_PATH.exists()
_model = None


def _load_model():
    """Build ResNet-18 with a 9-class head and load the saved weights."""
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    state = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _placeholder_probabilities(image):
    """Fake but repeatable probabilities. These are NOT real predictions."""
    seed = int(hashlib.md5(image.tobytes()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.ones(len(CLASSES)) * 0.3)
    return {c: float(p) for c, p in zip(CLASSES, raw)}


def predict_document(image):
    """Return {class: probability} for a PIL image."""
    global _model
    if USING_PLACEHOLDER:
        return _placeholder_probabilities(image)

    import torch
    from torchvision import transforms

    if _model is None:
        _model = _load_model()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(_model(tensor), dim=1)[0]
    return {c: float(p) for c, p in zip(CLASSES, probs)}