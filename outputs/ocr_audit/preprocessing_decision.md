# OCR Preprocessing Decision

**Project:** Neural Nexus Document Router
**Work package:** Preprocessing, OCR and baseline (branch `christa/ocr-baseline`)
**Evidence:** `outputs/ocr_audit/ocr_quality_audit.csv` and `outputs/ocr_audit/ocr_audit_summary.csv`

---

## 1. Decision

Use **raw-image OCR** as the baseline input for the text classifier, and keep the **Otsu pipeline** as a candidate for further evaluation.

The Otsu pipeline is grayscale conversion, 1.5x cubic upscaling and global Otsu thresholding, applied together. Because these steps were never tested separately, any difference between raw and Otsu output cannot be credited to thresholding alone.

The ten-document manual audit does not show that Otsu is generally better than raw OCR, so the simpler pipeline remains the default.

---

## 2. Evidence from the manual audit

**Method.** Ten training documents (seed 42) were read and scored for both raw and Otsu OCR text: two each from invoice, letter, form, resume and handwritten. Each text was scored 0 (unusable), 1 (partly readable) or 2 (mostly readable), judged against the original image, and one variant was preferred per document. No reference transcriptions exist, so these scores are readability judgements, not OCR accuracy, CER or WER.

**Overall result.**

| Measure | Raw | Otsu |
|---|---|---|
| Mean manual score | 1.4 | 1.4 |
| Documents scored 2 / 1 / 0 | 5 / 4 / 1 | 6 / 2 / 2 |
| Documents preferred | 4 | 6 |

Otsu scored higher on 1 document, the same on 8, and lower on 1. Four of the ten preferences were close calls: where outputs were effectively tied, raw was chosen by convention. The 6-to-4 preference split is therefore weak evidence.

**By document class.**

| Class | Mean raw | Mean Otsu | Observation |
|---|---|---|---|
| Form | 1.0 | 1.5 | Otsu recovered clearer printed headings (for example a fax cover sheet header). Table data stayed unusable in both. |
| Invoice | 1.5 | 1.5 | Mixed. Otsu recovered the total on one invoice; raw kept cleaner wording on the other. |
| Letter | 2.0 | 2.0 | Otsu recovered the letterhead and addressee on one letter; the other was a tie. |
| Resume | 2.0 | 2.0 | Both readable. Otsu was cleaner on field labels and dates, but years were misread in both. |
| Handwritten | 0.5 | 0.0 | Otsu was worse. It thickened ruled lines into junk text and lost phrases and the date that raw kept. One letter was legible to a human, but both outputs lost almost all of it. |

These results suggest that the effect of preprocessing depends on document type and image quality. A single global setting is therefore not supported.

**Automatic proxies.** Character count, word count and alphabetic-token ratio were recorded for each output. They are diagnostics only. More extracted text can mean more recovered content or more noise, and the alphabetic ratio is not a dictionary check. One handwritten document scored 0.82 on that ratio while its OCR was nearly unusable.

---

## 3. Limitations

- The audit has only ten documents and covers five of the nine classes. Budget, email, memo and questionnaire were not sampled.
- Scores and preferences are subjective and come from one reviewer.
- There are no reference transcriptions, so CER and WER were not calculated.
- Grayscale conversion, upscaling and thresholding were applied together, so their separate effects are unknown.
- Ten documents cannot represent the range of layouts, scan quality, text density and handwriting in the full dataset.
- The raw-versus-Otsu comparison used OCR text quality only. It was not tested against classification performance.

---

## 4. Additional training-only checks (proposed)

These checks have not yet been completed. The baseline in `outputs/baseline/` was trained on raw OCR only.

1. Run raw and Otsu OCR on the same training images with identical Tesseract settings.
2. Extend the comparison to a larger, reproducibly selected training sample covering all nine classes.
3. Record empty or very short outputs, character counts, word counts and alphabetic-token ratios for both variants.
4. Inspect representative successes and failures per class, noting missing text, misrecognition, reading-order problems and preprocessing noise.
5. If reliable transcriptions can be prepared for a small sample, calculate CER and WER on that sample.
6. Record the Tesseract version, preprocessing parameters, source image identifiers and seed.
7. Compare raw and Otsu text in the TF-IDF and Logistic Regression baseline, fitting on train only and judging on validation.

The test set must not be used to choose preprocessing parameters. It stays reserved until the final agreed evaluation. Validation data may be used only under the predefined model-selection procedure.

Preprocessing must also not depend on a document's true class. The class is unknown for incoming documents, so per-class findings are diagnostic only.

---

## 5. Implementation decision

- Keep raw OCR as the baseline and as the default pipeline.
- Keep Otsu available as an explicit, reproducible option (`--variant otsu` in `src/preprocessing/extract_ocr_dataset.py`).
- Do not switch to Otsu on the basis of six preferences out of ten when the mean scores are identical.
- Make the final choice after the broader training-only checks and the planned validation comparison. If the evidence stays inconclusive, keep raw OCR rather than claim an unsupported improvement.
