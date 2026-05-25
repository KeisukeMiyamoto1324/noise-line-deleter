from __future__ import annotations

from time import perf_counter

import torch
from transformers import AutoModel, AutoTokenizer


LINE_TOKEN = "<line>"
THRESHOLD = 0.6
TEXT = """
以下、Readme の日本語訳です。

CleanedWiki-jp
CleanedWiki-jp は、LLM の事前学習向けに整備された、日本語 Wikipedia のフィルター済みデータセットです。日本語 Wikipedia 記事の HTML をもとに作成され、Markdown に変換されたうえで、学習に適した形にフィルタリングされています。

このデータセットは、すべてを単なるプレーンテキストにするのではなく、有用な記事構造を保持しています。本文中の表は Markdown 表として保存され、数式は TeX 形式で保存されています。また、各行には日本十進分類法（NDC）のカテゴリと jReadability の難易度レベルも含まれており、利用者は分野や読解難易度に応じて学習データの比率を調整できます。

データセットの特徴
日本語 Wikipedia 記事の HTML ソースからクリーン化された本文を収録しています。
セクション見出し、段落、リスト、適切な表を Markdown 形式で保持しています。
数式を TeX 形式で保持しています。
参考文献、外部リンク、ナビゲーション的な内容、インフォボックス、ノイズの多い表、不適切なタイトル、低品質なテキストなど、学習に不要な要素を削除しています。
国立国会図書館の NDC Predictor によって予測された NDC メタデータを追加しています: https://lab.ndl.go.jp/service/ndc_predictor/
保持された各記事に対して、1〜6 の jReadability 難易度レベルを追加しています。
NDC ベースのサブセットを提供しており、利用者は広いトピックカテゴリごとの混合比率を調整できます。
"""


def main() -> None:
    device = torch.device("mps")
    tokenizer = AutoTokenizer.from_pretrained("MK0727/noise-line-remover-jp")
    model = AutoModel.from_pretrained("MK0727/noise-line-remover-jp", trust_remote_code=True).to(device)
    model.eval()

    lines = TEXT.split("\n")
    text = "".join(f"{LINE_TOKEN}{line}" for line in lines)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    start_time = perf_counter()
    with torch.no_grad():
        logits = model(**inputs).logits
    inference_time = perf_counter() - start_time
    probabilities = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()

    for line_number, (line, probability) in enumerate(zip(lines, probabilities, strict=True), start=1):
        label = "DELETE" if probability >= THRESHOLD else "KEEP"
        print(f"{line_number:02d} [{label}] {probability:.4f} {line}")

    print(f"Inference time: {inference_time:.4f} seconds")


if __name__ == "__main__":
    main()
