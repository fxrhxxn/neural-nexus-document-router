# Neural Nexus Document Router

**MA6700 — Applied AI and Machine Learning**  
**Assignment 3 — Group Project**

## Project Overview

The **Neural Nexus Document Router** is an AI-based document classification and routing system designed to automatically classify scanned or photographed documents and route them to the appropriate organisational department.

The project explores both **text-based** and **image-based** machine-learning approaches for document understanding. Scanned documents are processed using OCR to extract textual content, while deep-learning models analyse the visual structure of the document itself.

The system focuses on nine document categories and maps each predicted class to a higher-level routing destination.

---

## Problem Statement

Organisations may receive large numbers of scanned documents such as invoices, letters, forms, memos, questionnaires and resumes.

Manually inspecting and routing these documents is:

- repetitive;
- time-consuming;
- prone to human error;
- difficult to scale.

The project investigates how AI and machine learning can support automatic document classification while retaining a **manual-review mechanism for uncertain predictions**.

---

## Document Classes

The project uses the following nine document classes:

| Document Class | Routing Destination |
|---|---|
| Budget | Finance |
| Invoice | Finance |
| Email | Correspondence |
| Letter | Correspondence |
| Memo | Correspondence |
| Form | Administration |
| Questionnaire | Administration |
| Resume | Human Resources |
| Handwritten | Manual Review |

---

## Dataset

The primary dataset is a selected subset of the **RVL-CDIP document image dataset**, accessed through a Hugging Face mirror.

Nine document classes were selected for the project.

### Sampling

- **750 documents per class**
- **9 classes**
- **6,750 documents in total**
- Random seed: **42**

### Dataset Split

A stratified split was used so that each class remained evenly represented.

| Split | Documents | Documents per Class |
|---|---:|---:|
| Training | 4,725 | 525 |
| Validation | 1,008 | 112 |
| Test | 1,017 | 113 |
| **Total** | **6,750** | **750** |

The validation set is used during model development and threshold selection, while the test set is reserved for final evaluation.

---

## AI and Machine-Learning Approaches

The project investigates multiple approaches to document classification.

### OCR and Text Classification

OCR is used to extract text from scanned document images.

The traditional machine-learning baseline uses:

- Tesseract OCR
- TF-IDF feature extraction
- Logistic Regression

This provides a text-based baseline against which more advanced approaches can be compared.

### Image Classification

A **ResNet-18** convolutional neural network is used to classify documents using their visual appearance.

This allows the system to learn characteristics such as:

- page layout;
- structural patterns;
- text positioning;
- tables;
- formatting;
- visual document style.

### Confidence-Based Routing

Model confidence is used to support safer routing decisions.

Documents with sufficiently reliable predictions may be automatically routed, while uncertain documents can be directed to a **manual-review workflow**.

---

## Mauritian External Evaluation

In addition to the RVL-CDIP test set, a separate **Mauritian document evaluation set** is being prepared.

Its purpose is to observe how the final system behaves on documents that are more representative of a Mauritian organisational context.

The external set contains:

- public documents;
- anonymised or fictional mock documents;
- English-language documents;
- French-language documents;
- examples covering the project's document classes.

The Mauritian evaluation set is kept completely separate from the main RVL-CDIP dataset.

It is **not used for training, hyperparameter tuning, preprocessing selection or confidence-threshold selection**.

For multi-page documents, one representative page is selected before model predictions are generated.

---

## Project Structure

```text
neural-nexus-document-router/
|
|-- data/
|   |-- mauritian_test/
|   |   |-- documents/
|   |   |-- prepared/
|   |   `-- mauritian_test_metadata.csv
|
|-- models/
|
|-- outputs/
|   |-- figures/
|   |-- mauritian_evaluation/
|   |-- metrics/
|   `-- predictions/
|
|-- src/
|   |-- baseline/
|   |-- data/
|   |-- demo/
|   |-- evaluation/
|   |-- models/
|   |-- preprocessing/
|   `-- config.py
|
|-- tests/
|
|-- main.py
|-- requirements.txt
|-- README.md
`-- .gitignore
```

---

## Data Pipeline

The main dataset-related scripts are located in:

```text
src/data/
```

Important scripts include:

```text
inspect_rvl_cdip.py
sample_rvl_cdip.py
extract_sampled_images.py
create_dataset_split.py
validate_config.py
validate_final_dataset.py
prepare_mauritian_test.py
validate_mauritian_test.py
validate_prepared_mauritian_test.py
```

These scripts support dataset inspection, reproducible sampling, extraction, splitting and validation.

---

## Python Version

The project uses:

```text
Python 3.11
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/fxrhxxn/neural-nexus-document-router/tree/main
cd neural-nexus-document-router
```

### 2. Create a virtual environment

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Team

### Farhaan Hassam
**Team Leader / Data Pipeline**

Responsibilities include:

- overall project coordination;
- RVL-CDIP sampling;
- random-seed fixation;
- stratified dataset splitting;
- dataset validation;
- Mauritian external evaluation dataset preparation;
- project integration and submission management.

### Christa Mungada
**Preprocessing, OCR and Baseline**

Responsibilities include:

- image preprocessing;
- OCR experiments;
- OCR quality analysis;
- text extraction;
- TF-IDF feature generation;
- Logistic Regression baseline.

### Parisha Chooromonay
**Deep Learning and Architecture**

Responsibilities include:

- ResNet-18 document classification;
- deep-learning model development;
- model training and fine-tuning;
- investigation of text-based and multimodal architectures where feasible.

### Uddish Reetoo
**Evaluation, Thresholding and Demo**

Responsibilities include:

- model evaluation;
- confusion matrices;
- error analysis;
- confidence-threshold analysis;
- coverage-versus-accuracy analysis;
- manual-review logic;
- demonstration interface.

---

## Evaluation

The project considers both document-level classification and routing performance.

Important evaluation measures include:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Per-class performance
- Confusion matrices
- Invoice recall
- Routing accuracy
- Routing Macro F1
- Confidence-based coverage and accuracy

Final evaluation is performed using the held-out RVL-CDIP test set.

A separate qualitative evaluation is also conducted using the Mauritian document set.

---

## Limitations

The project has several limitations.

- Only nine RVL-CDIP document classes are used.
- The external Mauritian dataset is relatively small and should not be considered statistically representative of all Mauritian organisations.
- OCR quality may decrease for low-quality scans or handwritten documents.
- Document layout and terminology may differ from the RVL-CDIP training data.
- Confidence scores do not guarantee that a prediction is correct.
- The prototype focuses on classification and routing rather than a complete enterprise document-management workflow.

---

## Future Work

Possible extensions include:

- larger Mauritian document datasets;
- more document classes;
- improved handwriting recognition;
- multilingual document processing;
- transformer-based text classification;
- multimodal fusion of textual and visual features;
- stronger confidence calibration;
- document-level processing of complete multi-page files;
- integration with real document-management systems;
- continuous human-feedback and model-retraining workflows.

---

## Academic Context

This project was developed for the **MA6700 Applied AI and Machine Learning** module as a group assignment by **Team Neural Nexus**.