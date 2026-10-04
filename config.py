# config.py

"""
Shared configuration values for the Neural Nexus document routing project.

All team members should use the values defined here rather than creating
their own random seeds or dataset split ratios.
"""

RANDOM_SEED = 42

SAMPLES_PER_CLASS = 750

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

NUM_DOCUMENT_CLASSES = 9
NUM_ROUTING_OUTCOMES = 5


DOCUMENT_TO_DEPARTMENT = {
    "invoice": "Finance",
    "budget": "Finance",

    "letter": "Correspondence",
    "memo": "Correspondence",
    "email": "Correspondence",

    "resume": "HR",

    "form": "Administration",
    "questionnaire": "Administration",

    "handwritten": "Manual Review"
}