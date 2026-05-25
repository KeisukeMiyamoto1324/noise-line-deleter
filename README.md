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

## Quick Start

```python
import torch
from transformers import AutoModel, AutoTokenizer


LINE_TOKEN = "<line>"
THRESHOLD = 0.6
TEXT = """富士山は日本で最も高い山で、標高は3,776メートルである。
山頂付近は夏でも気温が低く、天候が急に変化することがある。
外部リンク: https://example.com/fuji
この記事は検証可能な参考文献が不足しています。
登山道は複数あり、利用者は体力や経験に応じて経路を選ぶ。
カテゴリ: 日本の山 | 火山 | 世界遺産"""

tokenizer = AutoTokenizer.from_pretrained("MK0727/noise-line-remover-jp")
model = AutoModel.from_pretrained("MK0727/noise-line-remover-jp", trust_remote_code=True)
model.eval()

lines = TEXT.split("\n")
text = "".join(f"{LINE_TOKEN}{line}" for line in lines)
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

probabilities = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()

for line_number, (line, probability) in enumerate(zip(lines, probabilities, strict=True), start=1):
    label = "DELETE" if probability >= THRESHOLD else "KEEP"
    print(f"{line_number:02d} [{label}] {probability:.4f} {line}")
```

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

Each input line has to be prefixed with a special line token, `<line>`. The model classifies the hidden state corresponding to each line token, so one forward pass can produce predictions for multiple lines.

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

## Inference Notes

For short texts, all lines can be processed in a single 4096-token window. For longer texts, split the document into overlapping windows and average probabilities for lines that appear in more than one window.

## Limitations

This model is specialized for Japanese web-text cleanup. It may perform poorly on other languages, highly structured documents, code, and tables.
