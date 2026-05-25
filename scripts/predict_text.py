from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from fineweb2_line_deleter.data import encode_line, trim_single_line, wrap_with_special_tokens
from fineweb2_line_deleter.model import LineNoiseModel, get_line_token_id


MODEL_REPO_ID = "MK0727/noise-line-remover-jp"
BASE_MODEL_NAME = "sbintuitions/modernbert-ja-130m"
FREEZE_UNTIL_LAYER = 10
MAX_LENGTH = 4096
LINE_OVERLAP = 4
THRESHOLD = 0.5
TEXT = """
LLM の pre-training 用コーパスである CleanedWiki-jp を公開しました。Wikipedia の全ての日本語記事をフィルタリングし、最終的に総トークン数は GPT-2 Tokenizer で 約2.2B になりました。

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

カラム

このデータセットには、次のカラムが含まれています。

id は Wikipedia 記事の識別子を示します。

url は 元の日本語 Wikipedia 記事の URL を示します。

title は 記事タイトルを示します。

ndc_code は 2 桁の予測 NDC コードを示します。カテゴリが利用できない場合は unknown になります。

ndc_category はコード付きの日本語 NDC カテゴリラベルを示します。

ndc_confidence は NDC 分類器が返した信頼度スコアを示します。

gpt2_token_count は text 内の GPT-2 BPE トークン数を示します。

jreadability_level は 1〜6 の jReadability レベルを示します。値が低いほど難しく、高いほど易しい文章を示します。

text は事前学習用にクリーン化された Markdown 本文です。

サブセット

NDC（日本十進分類法）は、図書館などで使われる日本の分類体系です。知識分野を 0〜9 の大きなカテゴリに分け、さらに細かい番号で主題を表します。CleanedWiki-jp のテキストはこのNDCによってサブセットにあらかじめ分割されており、データセットの利用者が目的に応じてテキストの混合比率を調整できるようになっています。なお、NDC分類の付与には国立国会図書館が提供する NDC Predicter を用いました。

利用可能なサブセットは、all、general_works、philosophy、history、social_sciences、natural_sciences、technology、industry、arts、language、literature です。

フィルタリング手順

このデータセットは、次の手順で作成されています。

各記事識別子について、最新のリビジョンを選択します。

一覧記事、曖昧さ回避ページ、リダイレクト、スラッシュを含むサブページなど、不適切なタイトルを除外します。

参考文献、注釈、外部リンク、関連項目、文献一覧、ナビゲーション的な付録など、不要な Wikipedia セクションを削除します。

インフォボックス、ナビボックス、参照マークアップ、スクリプト、スタイル、画像、その他の本文以外の要素など、不要な HTML クラスやタグを削除します。

Markdown 変換の前に、MediaWiki の数式ラッパーを標準的な TeX に変換します。

有用な本文を含む表は保持しつつ、不適切な表を削除します。

HTML からセクション見出し、段落、リスト、表を抽出し、Markdown に変換します。

空のセクション見出しと冒頭の読み仮名用の括弧を削除します。

Unicode を NFKC で正規化し、空白、参照記号、段落境界を正規化します。

短すぎるテキスト、段落や文が少なすぎるテキスト、日本語文字の比率が低いテキスト、記号やリストの量が過剰なテキスト、URL やウィキテキストの断片を含むテキスト、曖昧さ回避や参考文献のみの文章に見えるテキストを除外します。

クリーン化後のテキストが完全に重複しているものを削除します。

保持された各記事について、jReadability の難易度レベルを計算します。

保持された各記事について、NDC カテゴリを予測します。

最終的なクリーン済みテキストについて、GPT-2 BPE トークン数を数えます。

使用方法

from datasets import load_dataset

dataset = load_dataset("MK0727/CleanedWiki-jp", "all", split="train")
natural_sciences = load_dataset("MK0727/CleanedWiki-jp", "natural_sciences", split="train")
ライセンス

このデータセットは日本語 Wikipedia から派生したものです。元のテキストは Wikimedia の投稿者によって提供されており、Wikimedia のライセンス条件に従い、CC BY-SA 4.0 および GNU Free Documentation License のもとで利用できます。

CleanedWiki-jp は、CC BY-SA 4.0 のもとで配布されます。利用者は、元ライセンスの表示義務および継承義務に従う必要があります。
"""


@dataclass(frozen=True)
class InferenceWindow:
    input_ids: list[int]
    attention_mask: list[int]
    line_indices: list[int]


def main() -> None:
    device = torch.device("mps")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
    line_token_id = get_line_token_id(tokenizer)
    model_path = Path(hf_hub_download(repo_id=MODEL_REPO_ID, filename="model.pt"))
    model = LineNoiseModel(BASE_MODEL_NAME, line_token_id, len(tokenizer), FREEZE_UNTIL_LAYER)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    lines = TEXT.split("\n")
    start_time = perf_counter()
    probabilities = predict_line_probabilities(
        lines=lines,
        tokenizer=tokenizer,
        line_token_id=line_token_id,
        model=model,
        device=device,
        max_length=MAX_LENGTH,
        line_overlap=LINE_OVERLAP,
    )
    inference_time = perf_counter() - start_time

    for line_number, (line, probability) in enumerate(zip(lines, probabilities, strict=True), start=1):
        label = "DELETE" if probability >= THRESHOLD else "KEEP"
        print(f"{line_number:02d} [{label}] {probability:.4f} {line}")

    print(f"Inference time: {inference_time:.4f} seconds")


def predict_line_probabilities(
    lines: list[str],
    tokenizer: PreTrainedTokenizerBase,
    line_token_id: int,
    model: LineNoiseModel,
    device: torch.device,
    max_length: int,
    line_overlap: int,
) -> list[float]:
    windows = create_indexed_windows(
        lines=lines,
        tokenizer=tokenizer,
        line_token_id=line_token_id,
        max_length=max_length,
        line_overlap=line_overlap,
    )
    probability_lists: list[list[float]] = [[] for _ in lines]

    with torch.no_grad():
        for window in windows:
            input_ids = torch.tensor([window.input_ids], dtype=torch.long, device=device)
            attention_mask = torch.tensor([window.attention_mask], dtype=torch.long, device=device)
            outputs = model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs["logits"], dim=-1)[:, 1].detach().cpu().tolist()
            for line_index, probability in zip(window.line_indices, probabilities, strict=True):
                probability_lists[line_index].append(float(probability))

    return [sum(probabilities) / len(probabilities) for probabilities in probability_lists]


def create_indexed_windows(
    lines: list[str],
    tokenizer: PreTrainedTokenizerBase,
    line_token_id: int,
    max_length: int,
    line_overlap: int,
) -> list[InferenceWindow]:
    encoded_lines = [encode_line(line, tokenizer, line_token_id, max_length) for line in lines]
    windows: list[InferenceWindow] = []
    start_index = 0

    while start_index < len(encoded_lines):
        current_ids: list[int] = []
        line_indices: list[int] = []
        current_index = start_index

        while current_index < len(encoded_lines):
            candidate_ids = current_ids + encoded_lines[current_index]
            prepared_length = len(wrap_with_special_tokens(tokenizer, candidate_ids))
            if line_indices and prepared_length > max_length:
                break
            if prepared_length > max_length:
                current_ids = trim_single_line(encoded_lines[current_index], tokenizer, max_length)
                line_indices = [current_index]
                current_index += 1
                break
            current_ids = candidate_ids
            line_indices.append(current_index)
            current_index += 1

        input_ids = wrap_with_special_tokens(tokenizer, current_ids)
        windows.append(
            InferenceWindow(
                input_ids=input_ids,
                attention_mask=[1] * len(input_ids),
                line_indices=line_indices,
            )
        )

        if current_index >= len(encoded_lines):
            start_index = current_index
        else:
            retained_lines = min(line_overlap, len(line_indices) - 1)
            start_index = current_index - retained_lines

    return windows


if __name__ == "__main__":
    main()
