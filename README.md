---
language:
- ja
base_model: sbintuitions/modernbert-ja-310m
library_name: transformers
tags:
- text-classification
- line-classification
- japanese
- corpus-cleaning
- modernbert
---

# FineWeb2 Line Deleter

FineWeb2 Line Deleter is a Japanese line-level noise classifier for corpus cleanup. Given a multi-line text, it predicts whether each line should be kept or deleted.

This model is fine-tuned from [`sbintuitions/modernbert-ja-310m`](https://huggingface.co/sbintuitions/modernbert-ja-310m). The model uses ModernBERT hidden states at special line-token positions and applies a binary classifier to each line.

## Intended Use

This model is intended for preprocessing Japanese web corpora before language model training. It is useful when a dataset contains lines such as boilerplate text, navigation fragments, repeated links, low-value fragments, or other noisy content that should be removed while keeping useful body text.

The output is a delete probability for each line. A typical workflow is:

1. Split a document into lines.
2. Run line-level prediction.
3. Delete lines whose probability is above a chosen threshold.
4. Join the remaining lines back into cleaned text.

The recommended threshold is `0.6`, but downstream users may tune it depending on whether they prefer higher precision or higher recall.

## Model Details

- **Base model:** [`sbintuitions/modernbert-ja-310m`](https://huggingface.co/sbintuitions/modernbert-ja-310m)
- **Task:** binary line-level classification
- **Positive label:** line should be deleted
- **Negative label:** line should be kept
- **Input length:** up to 4096 tokens per window
- **Long documents:** split into overlapping line windows, then duplicate predictions are averaged
- **Architecture:** ModernBERT encoder plus a classification head over line-token hidden states

Each input line is prefixed with a special line token. The model classifies the hidden state corresponding to each line token, so one forward pass can produce predictions for multiple lines.

## Training Data

The model was trained on `MK0727/line-noise-label`, a Japanese line-level dataset with delete labels.
Training used 4096-token windows, 3 epochs.

## Performance

Final test metrics from the same run:

| Metric | Value |
| --- | ---: |
| Test precision for delete lines | 0.8041 |
| Test recall for delete lines | 0.7572 |
| Test F1 for delete lines | 0.7799 |
| Test accuracy | 0.8267 |
| Test macro F1 | 0.8185 |

## Inference Notes

For short texts, all lines can be processed in a single 4096-token window. For longer texts, split the document into overlapping windows and average probabilities for lines that appear in more than one window.

The included inference script uses overlapping windows so that boundary lines still receive context from neighboring lines. This is important because noisy lines are often easier to identify from surrounding structure, not only from the line itself.

## Limitations

This model is specialized for Japanese web-text cleanup. It may perform poorly on other languages, highly structured documents, code, and tables.
