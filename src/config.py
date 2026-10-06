"""
Shared configuration for the Neural Nexus Document Router project.

This file contains values that must remain consistent across the project,
including:

- random seed
- dataset sampling size
- dataset split ratios
- RVL-CDIP class labels
- selected document classes
- document-to-routing-outcome mapping

All team members should import these values rather than redefining them
inside individual scripts.
"""


# ============================================================
# Reproducibility
# ============================================================

# Fixed random seed used throughout the project.
# This helps make sampling, splitting and experiments reproducible.
RANDOM_SEED = 42


# ============================================================
# Dataset sampling
# ============================================================

# Number of documents to sample from each selected document class.
SAMPLES_PER_CLASS = 750

# We selected 9 document types from RVL-CDIP.
NUM_SELECTED_CLASSES = 9

# Expected total sampled dataset size:
# 750 documents × 9 classes = 6,750 documents.
TOTAL_SAMPLES = SAMPLES_PER_CLASS * NUM_SELECTED_CLASSES


# ============================================================
# Dataset split ratios
# ============================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# RVL-CDIP class labels
# ============================================================

# IMPORTANT:
# These IDs were verified directly from the streamed
# Chan814/rvl-cdip Hugging Face dataset.
#
# Do not assume that another RVL-CDIP implementation will use
# the same numeric ordering.

RVL_CDIP_LABELS = {
    0: "advertisement",
    1: "budget",
    2: "email",
    3: "file folder",
    4: "form",
    5: "handwritten",
    6: "invoice",
    7: "letter",
    8: "memo",
    9: "news article",
    10: "presentation",
    11: "questionnaire",
    12: "resume",
    13: "scientific publication",
    14: "scientific report",
    15: "specification",
}


# ============================================================
# Selected document classes
# ============================================================

# These are the 9 document types selected for the project.

SELECTED_CLASS_IDS = {
    1: "budget",
    2: "email",
    4: "form",
    5: "handwritten",
    6: "invoice",
    7: "letter",
    8: "memo",
    11: "questionnaire",
    12: "resume",
}


# ============================================================
# Document type → routing outcome
# ============================================================

# The application's main purpose is routing documents.
#
# NOTE:
# "Manual Review" is a routing outcome rather than a department.
# This is why the variable is called DOCUMENT_TO_ROUTING_OUTCOME
# rather than DOCUMENT_TO_DEPARTMENT.

DOCUMENT_TO_ROUTING_OUTCOME = {
    "budget": "Finance",
    "invoice": "Finance",

    "email": "Correspondence",
    "letter": "Correspondence",
    "memo": "Correspondence",

    "resume": "HR",

    "form": "Administration",
    "questionnaire": "Administration",

    "handwritten": "Manual Review",
}


# ============================================================
# Routing outcomes
# ============================================================

ROUTING_OUTCOMES = [
    "Finance",
    "Correspondence",
    "HR",
    "Administration",
    "Manual Review",
]