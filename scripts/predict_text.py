from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from fineweb2_line_deleter.data import encode_line, wrap_with_special_tokens
from fineweb2_line_deleter.model import LineNoiseModel, get_line_token_id


MODEL_REPO_ID = "MK0727/noise-line-remover-jp"
BASE_MODEL_NAME = "sbintuitions/modernbert-ja-130m"
FREEZE_UNTIL_LAYER = 10
MAX_LENGTH = 4096
THRESHOLD = 0.5
TEXT = """
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
    line_token_id = get_line_token_id(tokenizer)
    model_path = Path(hf_hub_download(repo_id=MODEL_REPO_ID, filename="model.pt"))
    model = LineNoiseModel(BASE_MODEL_NAME, line_token_id, len(tokenizer), FREEZE_UNTIL_LAYER)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    input_ids: list[int] = []
    for line in TEXT.split("\n"):
        input_ids.extend(encode_line(line, tokenizer, line_token_id, MAX_LENGTH))
    input_ids = wrap_with_special_tokens(tokenizer, input_ids)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([[1] * len(input_ids)], dtype=torch.long, device=device)

    with torch.no_grad():
        start_time = perf_counter()
        outputs = model(input_tensor, attention_mask)
        inference_time = perf_counter() - start_time
        probabilities = torch.softmax(outputs["logits"], dim=-1)[:, 1].detach().cpu().tolist()

    for line_number, (line, probability) in enumerate(zip(TEXT.split("\n"), probabilities, strict=True), start=1):
        label = "DELETE" if probability >= THRESHOLD else "KEEP"
        print(f"{line_number:02d} [{label}] {probability:.4f} {line}")

    print(f"Inference time: {inference_time:.4f} seconds")


if __name__ == "__main__":
    main()
