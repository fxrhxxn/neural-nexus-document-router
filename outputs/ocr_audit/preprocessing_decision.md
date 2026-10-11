#### OCR Preprocessing Decision





**1. Decision**



Retain raw-image OCR as the baseline and evaluate the existing Otsu preprocessing pipeline as an alternative. The current ten-document manual audit does not provide sufficient evidence to conclude that Otsu is consistently better.



The Otsu pipeline includes grayscale conversion, 1.5x upscaling and Otsu thresholding. Therefore, any observed differences cannot be attributed to thresholding alone.





**2. Evidence from the manual audit**



The audit covers ten documents from five of the nine selected document classes: invoice, letter, form, resume and handwritten.



The recorded character counts show mixed results. Otsu preprocessing produces more characters for some documents and fewer for others. This suggests that preprocessing changes the extracted text, but character counts alone do not establish whether the text is more accurate.



The audit also records word counts, alphabetic-token ratios and manual quality judgements. These provide preliminary qualitative evidence, not objective OCR accuracy measurements.





**3. Limitations**



\- The audit contains only ten documents and does not cover budget, email, memo or questionnaire classes.

\- Manual quality scores are subjective.

\- No verified reference transcriptions were used to calculate character error rate (CER) or word error rate (WER).

\- The Otsu pipeline combines grayscale conversion, upscaling and thresholding, so their individual effects cannot be isolated.

\- Results may differ across document classes, handwriting, layouts, image quality and text density.

\- The small audit is exploratory and cannot establish general superiority of either method.





**4. Additional training-only checks**



1\. Verify that raw and Otsu OCR use the same source images and OCR engine settings.

2\. Extend the comparison to a broader sample from the training split, covering all nine selected classes.

3\. Record empty outputs, unusually short outputs, character counts, word counts and alphabetic-token ratios.

4\. Manually inspect representative failures from each class and record the reasons for the final choice.

5\. If reliable reference transcriptions can be prepared, calculate CER and WER on that explicitly transcribed training sample.

6\. Record the OCR engine version, preprocessing parameters, source image identifiers and random seed.

7\. Cache extraction results so interrupted runs can resume without repeating completed work.



These checks must use training data only. Do not use the reserved test set to choose preprocessing parameters. Keep validation data separate from training-only diagnostics and use it for the planned model-selection and evaluation procedure.





**5. Implementation decision**



Preserve raw OCR as the baseline and retain Otsu as a candidate for further evaluation. Make preprocessing settings explicit and reproducible.



Choose the final configuration only after reviewing the broader evidence. If the results remain inconclusive, retain the simpler raw pipeline rather than claiming an unsupported improvement.



