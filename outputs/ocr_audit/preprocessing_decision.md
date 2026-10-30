\# OCR Preprocessing Decision



\## 1. Decision



Retain raw-image OCR as the baseline and evaluate the existing Otsu preprocessing pipeline as an alternative. The ten-document manual audit does not establish that Otsu preprocessing is generally superior to raw OCR.



The Otsu pipeline combines grayscale conversion, 1.5× upscaling and Otsu thresholding. Consequently, observed differences cannot be attributed to thresholding alone.



\## 2. Evidence from the Manual Audit



The audit covers ten training documents from five of the nine selected classes: invoice, letter, form, resume and handwritten.



The mean manual quality score was 1.4 for both raw OCR and Otsu OCR, using a scale from 0 (unusable) to 2 (mostly readable). Otsu was preferred for six documents and raw OCR for four. However, the manual scores were equal for eight documents, so these preferences reflect differences in the content recovered, errors introduced and readability of the extracted text.



Otsu recovered additional printed headings and text in some invoice, letter and form examples. In contrast, it performed poorly on the handwritten examples, where it lost useful content and introduced noise from ruled lines. These results suggest that preprocessing effectiveness may depend on document type and image characteristics.



Character counts, word counts and alphabetic-token ratios provide useful diagnostic information, but they cannot establish OCR accuracy because additional extracted text may also contain errors or noise.



\## 3. Limitations



\- The audit contains only ten documents and does not cover the budget, email, memo or questionnaire classes.

\- Manual quality scores and preferred-variant judgements are subjective.

\- No verified reference transcriptions were used to calculate character error rate (CER) or word error rate (WER).

\- Grayscale conversion, upscaling and thresholding were applied together, so their individual contributions cannot be isolated.

\- The small sample may not represent the range of layouts, image quality, text density and handwriting in the dataset.

\- The results are exploratory and do not support a general claim that either preprocessing method is superior.



\## 4. Additional Training-Only Checks



1\. Compare raw and Otsu OCR on the same source images using identical OCR engine settings.

2\. Extend the comparison to a larger, reproducibly selected sample from the training split, covering all nine document classes.

3\. Record empty or unusually short outputs, character counts, word counts and alphabetic-token ratios for both variants.

4\. Manually inspect representative successes and failures from each class, documenting missing text, misrecognition, reading-order problems and preprocessing noise.

5\. If reliable reference transcriptions can be prepared, calculate CER and WER for the explicitly transcribed training sample.

6\. Record the OCR engine and version, preprocessing parameters, source image identifiers and random seed.

7\. Cache OCR outputs and record processing status so interrupted extraction runs can resume without repeating completed work.



These checks must not use the reserved test set to choose preprocessing parameters. Keep the test set untouched until the planned final evaluation. Use validation data only according to the predefined model-selection and evaluation procedure.



\## 5. Implementation Decision



Preserve raw OCR as the baseline and retain Otsu as a candidate for further evaluation. Keep preprocessing configurations explicit and reproducible.



Do not select Otsu solely because it was preferred for six of the ten audited documents. Make the final decision after reviewing the broader training-only evidence and planned validation results. If the evidence remains inconclusive, retain the simpler raw OCR pipeline as the default rather than claiming an unsupported improvement.

