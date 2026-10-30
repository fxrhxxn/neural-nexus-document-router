# Neural Nexus Document Router

**MA6700 — Applied AI and Machine Learning**  
**Assignment 3 — Group Project**  
**Team: Neural Nexus**

## Project Overview

The **Neural Nexus Document Router** is an AI-based system for automatically classifying scanned documents and routing them to the appropriate organisational department.

The project evaluates two approaches:

- **OCR-based text classification** using Tesseract, TF-IDF and Logistic Regression.
- **Image classification** using a fine-tuned ResNet-18.

The final Streamlit demonstration uses the **ResNet-18 image-classification pipeline** together with confidence-based routing and manual review.

---

## Problem Statement

Organisations may receive large numbers of scanned documents such as invoices, letters, forms, memos, questionnaires and resumes. Manual inspection and routing can be repetitive, time-consuming, prone to error and difficult to scale.

This project investigates how AI and machine learning can support automatic document classification while retaining a **manual-review mechanism for uncertain predictions**.

---

## Document Classes and Routing

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

Handwritten documents are always sent to **Manual Review**.

---

## Dataset

The project uses a selected subset of the **RVL-CDIP document image dataset**, accessed through a Hugging Face mirror.

### Sampling

- 9 document classes
- 750 documents per class
- 6,750 documents in total
- Fixed random seed: **42**

### Dataset Split

| Split | Documents | Documents per Class |
|---|---:|---:|
| Training | 4,725 | 525 |
| Validation | 1,008 | 112 |
| Test | 1,017 | 113 |
| **Total** | **6,750** | **750** |

The validation set was used for model development, checkpoint selection and threshold selection. The held-out test set was used only after the final model configuration and routing threshold were frozen.

---

## Mauritian External Evaluation Set

A separate **Mauritian-oriented evaluation set** was prepared to observe how the final system behaves on more locally representative documents.

The set contains **40 documents**:

| Class | Count |
|---|---:|
| Budget | 4 |
| Email | 4 |
| Form | 5 |
| Handwritten | 3 |
| Invoice | 6 |
| Letter | 5 |
| Memo | 5 |
| Questionnaire | 5 |
| Resume | 3 |

Language distribution:

- **26 English documents**
- **14 French documents**

The set was kept separate from the RVL-CDIP training, validation and test data and was **not used for training, model selection, preprocessing selection or threshold tuning**.

For multi-page PDFs, one representative page was selected before prediction and converted to RGB PNG at **200 DPI**. Because the set is small and manually curated, it is treated as a **qualitative external evaluation** rather than a statistically representative benchmark.

---

## Approaches Implemented

### 1. OCR + TF-IDF + Logistic Regression Baseline

The text-based baseline uses:

1. **Tesseract OCR** to extract text.
2. **TF-IDF** to create numerical text features.
3. **Logistic Regression** to classify the document.
4. Class-to-route mapping to determine the organisational destination.

The OCR study compared raw-image OCR with an Otsu-based preprocessing pipeline. The small manual audit produced the same average readability score for both approaches, so **raw OCR was retained as the baseline**.

### Baseline Validation Results

| Metric | Result |
|---|---:|
| 9-class accuracy | 0.7808 |
| 9-class macro-F1 | 0.7858 |
| Routing accuracy | 0.8323 |
| Routing macro-F1 | 0.8288 |

---

### 2. ResNet-18 Image Classifier

A pretrained **ResNet-18** was adapted using transfer learning.

The final configuration used:

- ImageNet-pretrained weights
- RGB input
- 224 × 224 image size
- ImageNet normalisation
- Partial fine-tuning of `layer4` and the final classifier
- Adam optimiser
- Cross-Entropy Loss
- Learning rate: **0.0001**
- Batch size: **16**
- Light random rotation of approximately **±2°**

### ResNet Experimental Progression

| Experiment | Best Epoch | 9-Class Accuracy | 9-Class Macro-F1 | Routing Accuracy | Routing Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Frozen ResNet-18, 5 epochs | 5 | 0.5665 | 0.5598 | 0.6786 | 0.6800 |
| Frozen ResNet-18, 10 epochs | 8 | 0.5774 | 0.5663 | 0.7093 | 0.7148 |
| Partial fine-tuning | 8 | 0.7837 | 0.7831 | 0.8343 | 0.8421 |
| Partial fine-tuning + augmentation | 10 | 0.7907 | 0.7891 | 0.8393 | **0.8445** |

The partially fine-tuned model with light augmentation achieved the strongest validation performance and was selected as the final ResNet configuration.

### Final Held-Out Test Results

| Metric | Result |
|---|---:|
| 9-class accuracy | 0.7768 |
| 9-class macro-F1 | 0.7759 |
| Routing accuracy | 0.8348 |
| Routing macro-F1 | 0.8373 |
| Invoice recall | 0.726 |

---

## Confidence-Based Routing

The final system uses **route confidence** rather than document-class probability alone.

Route confidence is the sum of the probabilities of all document classes belonging to the predicted department.

For example:

```text
Correspondence = email + letter + memo
```

Routing rule:

```text
route confidence >= 0.95  -> AUTO_ROUTE
route confidence < 0.95   -> MANUAL_REVIEW
predicted handwritten     -> MANUAL_REVIEW
```

The threshold **τ = 0.95** was selected using the validation set only.

At this threshold:

- Validation auto-routed accuracy: **95.22%**
- Test auto-routed coverage: **56.54%**
- Test auto-routed accuracy: **94.96%**

The threshold was not changed after observing the test-set result.

---

## Streamlit Demonstration

The project includes a Streamlit interface for demonstrating the final ResNet routing pipeline.

Users can upload:

- PNG
- JPG / JPEG
- PDF — first page only

The interface displays:

- Uploaded document
- Predicted document type
- Predicted department
- Route confidence
- Automatic-routing or manual-review decision
- Class-probability chart

### Run the Demo

From the project root:

```powershell
python -m streamlit run src/demo/app.py
```

Run Version 2 from the project root with:

```powershell
streamlit run src/demo/app_v2.py
```

---

## Model Checkpoint

The final ResNet checkpoint is excluded from Git because model files are ignored by `.gitignore`.

To use the real trained model, place the checkpoint at:

```text
models/best_resnet18.pt
```

If the checkpoint is missing, the demo falls back to a clearly labelled **placeholder mode** and does not produce real model predictions.

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
|   |-- .gitkeep
|   `-- best_resnet18.pt
|
|-- outputs/
|   |-- baseline/
|   |-- deep_learning/
|   |-- figures/
|   |-- mauritian_evaluation/
|   |-- metrics/
|   |-- ocr_audit/
|   |-- predictions/
|   `-- preprocessing_tests/
|
|-- src/
|   |-- baseline/
|   |-- data/
|   |-- demo/
|   |   |-- app.py
|   |   `-- predictor.py
|   |-- evaluation/
|   |   |-- error_analysis.py
|   |   |-- metrics.py
|   |   |-- probs_to_predictions.py
|   |   |-- run_evaluation.py
|   |   `-- thresholding.py
|   |-- models/
|   |-- preprocessing/
|   `-- config.py
|
|-- tests/
|-- requirements.txt
|-- README.md
`-- .gitignore
```

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/fxrhxxn/neural-nexus-document-router.git
cd neural-nexus-document-router
```

### 2. Create a Python 3.11 Virtual Environment

#### Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the Final ResNet Checkpoint

Place:

```text
best_resnet18.pt
```

inside:

```text
models/
```

### 5. Run the Streamlit Demo

```bash
python -m streamlit run src/demo/app.py
```

---

## Recorded Baseline Configuration and Software Versions

The following configuration was recorded for the **OCR + TF-IDF + Logistic Regression baseline**.

| Setting | Recorded Value |
|---|---|
| OCR variant | Raw |
| TF-IDF lowercase | True |
| TF-IDF accent handling | Unicode accent stripping |
| TF-IDF n-grams | (1,2) — unigrams and bigrams |
| Maximum TF-IDF features | 100,000 |
| Sublinear term frequency | True |
| Logistic Regression max iterations | 2000 |
| Random seed / random state | 42 |
| Python | 3.11.1 |
| Scikit-learn | 1.9.1 |
| SciPy | 1.18.1 |
| NumPy | 2.5.3 |
| Pandas | 3.0.6 |

---

## Evaluation

Evaluation was performed at both document-class and routing level.

### Document-Level Metrics

- Accuracy
- Macro-F1
- Per-class precision and recall
- Confusion matrix

### Routing-Level Metrics

- Routing accuracy
- Routing macro-F1
- Route confidence
- Coverage versus accuracy
- Manual-review rate
- Silent routing errors
- Invoice recall

The project distinguishes classification errors from routing errors because an incorrect document class may still map to the correct department.

---

## Key Findings

- The OCR + TF-IDF baseline achieved **0.8288 routing macro-F1**.
- Frozen ImageNet features alone were insufficient.
- Partial ResNet fine-tuning produced the largest improvement.
- Light augmentation produced a smaller additional improvement.
- The final ResNet achieved **0.8445 validation routing macro-F1** and **0.8373 test routing macro-F1**.
- Confidence thresholding improved the reliability of automatically routed documents but reduced coverage.
- High-confidence routing errors still occurred, showing that confidence alone cannot guarantee correctness.
- Visually similar classes such as forms, questionnaires, letters and memos remained challenging.
- The current Streamlit demo uses the **image-based ResNet pipeline only**.

---

## Limitations

- Only nine RVL-CDIP classes are used.
- The ResNet model showed signs of overfitting.
- The final demo is image-only and does not combine the OCR baseline with the ResNet model.
- Visually similar classes remain difficult to distinguish.
- Confidence scores do not guarantee correctness.
- OCR quality can degrade on poor-quality or handwritten scans.
- The OCR audit used only 10 documents across 5 of the 9 classes.
- The Mauritian evaluation set is small and not statistically representative.
- The Streamlit demo processes only the first page of PDFs.
- The trained ResNet checkpoint is not stored in the repository.

---

## Future Work

Possible improvements include:

- Combining the ResNet image classifier with the OCR-based text classifier
- Transformer-based text classification using models such as DistilBERT
- Multimodal fusion of textual and visual features
- Stronger regularisation and augmentation
- Larger English and French Mauritian evaluation datasets
- Confidence calibration
- Multi-page document processing
- Improved handwriting recognition
- Integration with enterprise document-management systems

---

## Team Contributions

### Farhaan Hassam
**Team Leader / Data Pipeline**

- Project coordination
- RVL-CDIP sampling
- Random-seed fixation
- Stratified train/validation/test splitting
- Dataset validation
- Mauritian external evaluation set
- Project integration and submission management

### Christa Mungada
**Preprocessing, OCR and Baseline**

- Image preprocessing
- OCR experiments
- OCR quality analysis
- OCR text extraction
- TF-IDF feature generation
- Logistic Regression baseline

### Parisha Chooromonay
**Deep Learning**

- ResNet-18 image-classification pipeline
- Model training
- Frozen-backbone experiments
- Partial fine-tuning
- Augmentation experiments
- Final checkpoint selection

### Uddish Reetoo
**Evaluation, Thresholding and Demo**

- Model evaluation
- Confusion matrices
- Error analysis
- Confidence-threshold analysis
- Coverage-versus-accuracy analysis
- Manual-review logic
- Streamlit demonstration interface

---

## Academic Context

This project was developed for the **MA6700 Applied AI and Machine Learning** module as a group assignment by **Team Neural Nexus**.

---

## References

Harley, A.W., Ufkes, A. and Derpanis, K.G. (2015) ‘Evaluation of deep convolutional nets for document image classification and retrieval’, *Proceedings of the 13th International Conference on Document Analysis and Recognition (ICDAR)*.

He, K., Zhang, X., Ren, S. and Sun, J. (2016) ‘Deep residual learning for image recognition’, *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778.

Smith, R. (2007) ‘An overview of the Tesseract OCR engine’, *Proceedings of the Ninth International Conference on Document Analysis and Recognition*, pp. 629–633.

Salton, G. and Buckley, C. (1988) ‘Term-weighting approaches in automatic text retrieval’, *Information Processing & Management*, 24(5), pp. 513–523.

Chan814 (n.d.) *RVL-CDIP dataset*. Hugging Face Datasets.
