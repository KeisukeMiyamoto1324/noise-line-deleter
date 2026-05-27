---
language:
- ja
base_model: sbintuitions/modernbert-ja-130m
library_name: transformers
tags:
- text-classification
- line-classification
- japanese
- corpus-cleaning
- modernbert
---

# corpus-refiner-jp

corpus-refiner-jp is a Japanese line-level keep classifier for corpus cleanup. Given a multi-line text, it predicts whether each line should be kept.

This model is fine-tuned from [`sbintuitions/modernbert-ja-130m`](https://huggingface.co/sbintuitions/modernbert-ja-130m). The model uses ModernBERT hidden states at special line-token positions and applies a binary classifier to each line.

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


def main() -> None:
    # ---------------------------------------------------------
    # Load tokenizer and AutoModel-compatible line classifier.
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained("MK0727/corpus-refiner-jp")
    model = AutoModel.from_pretrained("MK0727/corpus-refiner-jp", trust_remote_code=True)
    model.eval()

    # ---------------------------------------------------------
    # Add the line marker before each input line.
    # ---------------------------------------------------------
    lines = TEXT.split("\n")
    text = "".join(f"{LINE_TOKEN}{line}" for line in lines)
    inputs = tokenizer(text, return_tensors="pt")

    # ---------------------------------------------------------
    # Predict the keep probability for each line marker.
    # ---------------------------------------------------------
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()

    # ---------------------------------------------------------
    # Print each line with its predicted label and probability.
    # ---------------------------------------------------------
    for line_number, (line, probability) in enumerate(zip(lines, probabilities, strict=True), start=1):
        label = "KEEP" if probability >= THRESHOLD else "DELETE"
        print(f"{line_number:02d} [{label:<6}] {probability:.4f} {line}")


if __name__ == "__main__":
    main()
```

Example output:

```
01 [KEEP  ] 0.9673 富士山は日本で最も高い山で、標高は3,776メートルである。
02 [KEEP  ] 0.9888 山頂付近は夏でも気温が低く、天候が急に変化することがある。
03 [DELETE] 0.0240 外部リンク: https://example.com/fuji
04 [DELETE] 0.0817 この記事は検証可能な参考文献が不足しています。
05 [KEEP  ] 0.8815 登山道は複数あり、利用者は体力や経験に応じて経路を選ぶ。
06 [DELETE] 0.0447 カテゴリ: 日本の山 | 火山 | 世界遺産
```

## Intended Use

This model is intended for preprocessing Japanese web corpora before language model training. It is useful when a dataset contains boilerplate text, navigation fragments, repeated links, low-value fragments, or other content that should be removed while preserving useful body text.

The output is a keep probability for each line. A typical workflow is:

1. Split a document into lines.
2. Run line-level prediction.
3. Keep lines whose probability is above a chosen threshold.
4. Join the kept lines back into cleaned text.

The recommended keep threshold should be chosen after evaluating the trained checkpoint on held-out data.

## Model Details

- **Base model:** [`sbintuitions/modernbert-ja-130m`](https://huggingface.co/sbintuitions/modernbert-ja-130m)
- **Task:** binary line-level classification
- **Positive label:** line should be kept
- **Negative label:** line should be deleted
- **Label mapping:** `0=DELETE`, `1=KEEP`
- **Input length:** up to 4096 tokens per window
- **Long documents:** split into overlapping line windows, then duplicate predictions are averaged
- **Architecture:** ModernBERT encoder plus a classification head over line-token hidden states

Each input line has to be prefixed with a special line token, `<line>`. The model classifies the hidden state corresponding to each line token, so one forward pass can produce predictions for multiple lines.

## Training Data

The model is trained on `MK0727/noise-line-label-jp`, a Japanese line-level dataset with `lines_to_keep` annotations.

Training uses a generated train/valid/test split from the dataset's `train` split.

## Performance

Metrics should be regenerated after training the new keep-based checkpoint. The training script reports keep-positive metrics such as `precision_keep`, `recall_keep`, and `f1_keep`.

## Inference Notes

For short texts, all lines can be processed in a single 4096-token window. For longer texts, split the document into overlapping windows and average probabilities for lines that appear in more than one window.

## Limitations

This model is specialized for Japanese web-text cleanup. It may perform poorly on other languages, highly structured documents, code, and tables.
