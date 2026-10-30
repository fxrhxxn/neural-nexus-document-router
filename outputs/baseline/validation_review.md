# Validation Review — OCR Document Classification Baseline

## Objective

Evaluate the TF-IDF + Logistic Regression baseline using the validation split only. The model predicts nine document classes, which are subsequently mapped to five routing outcomes.

## Validation Results

- Training records: 4,725
- Validation records: 1,008
- Document-class accuracy: 78.08%
- Document-class macro F1: 0.7858
- Routing accuracy: 83.23%
- Routing macro F1: 0.8288
- Incorrect document-class predictions: 221
- Incorrect routing predictions: 169
- Document-class errors that preserve the correct route: 52

## OCR Extraction Quality

- Training OCR records: 4,725; 79 had empty extracted text.
- Validation OCR records: 1,008; 22 had empty extracted text.
- Failed extractions reported: 0 in both splits.

Empty OCR text is not the same as an extraction failure. Empty-text cases should be considered when interpreting model performance.

## Error Analysis

Correspondence and HR routing performed strongly in the current validation results. Administration routing recall was 69.20%, while Manual Review precision was 54.10%. These results indicate areas for further investigation.

Some document-class mistakes still preserve the correct routing outcome. The routing metrics provide an additional view, but they do not establish real-world operational performance.

## 97% Validation Accuracy Tripwire

The tripwire was not triggered. Document-class validation accuracy was 78.08%, below the 97% threshold. No tripwire investigation was required by the stated threshold.

## Test-Set Protection

The test split was not used to train, select, tune or evaluate this baseline. All results in this review are from the training and validation splits. The test set must remain reserved until modelling decisions are frozen and final evaluation is authorised.

## Limitations and Next Research Questions

The baseline uses OCR text with TF-IDF and Logistic Regression. It does not establish that raw OCR is optimal, nor does it demonstrate performance on new documents from a different source. The manual preprocessing audit was small and subjective.

Future experiments may compare preprocessing variants and alternative models using training data for fitting and validation data for model selection, without using the test set.

## Reproducibility

The baseline configuration, saved model pipeline, validation predictions, classification report, confusion matrices and error-analysis reports are stored under `outputs/baseline/`.

The OCR extraction pipeline is stored at `src/preprocessing/extract_ocr_dataset.py`. Validation analysis is stored at `src/baseline/analyze_validation.py`.
