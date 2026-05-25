import torch
from transformers import AutoModel, AutoTokenizer


LINE_TOKEN = "<line>"
THRESHOLD = 0.6
TEXT = """
富士山は日本で最も高い山で、標高は3,776メートルである。
山頂付近は夏でも気温が低く、天候が急に変化することがある。
外部リンク: https://example.com/fuji
この記事は検証可能な参考文献が不足しています。
登山道は複数あり、利用者は体力や経験に応じて経路を選ぶ。
カテゴリ: 日本の山 | 火山 | 世界遺産
"""


def main() -> None:
    # ---------------------------------------------------------
    # Load tokenizer and AutoModel-compatible line classifier.
    # ---------------------------------------------------------
    device = torch.device("mps")
    tokenizer = AutoTokenizer.from_pretrained("MK0727/noise-line-remover-jp")
    model = AutoModel.from_pretrained("MK0727/noise-line-remover-jp", trust_remote_code=True).to(device)
    model.eval()

    # ---------------------------------------------------------
    # Add the line marker before each input line.
    # ---------------------------------------------------------
    lines = TEXT.split("\n")
    text = "".join(f"{LINE_TOKEN}{line}" for line in lines)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # ---------------------------------------------------------
    # Predict the deletion probability for each line marker.
    # ---------------------------------------------------------
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()

    # ---------------------------------------------------------
    # Print each line with its predicted label and probability.
    # ---------------------------------------------------------
    for line_number, (line, probability) in enumerate(zip(lines, probabilities, strict=True), start=1):
        label = "DELETE" if probability >= THRESHOLD else "KEEP"
        print(f"{line_number:02d} [{label}] {probability:.4f} {line}")


if __name__ == "__main__":
    main()
