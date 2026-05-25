from __future__ import annotations

from time import perf_counter

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


MODEL_REPO_ID = "MK0727/noise-line-remover-jp"
LINE_TOKEN = "<line>"
THRESHOLD = 0.5
TEXT = """

"""


def main() -> None:
    device = torch.device("mps")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
    model = AutoModel.from_pretrained(MODEL_REPO_ID, trust_remote_code=True).to(device)
    model.eval()

    lines = TEXT.split("\n")
    start_time = perf_counter()
    probabilities = predict_line_probabilities(
        lines=lines,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    inference_time = perf_counter() - start_time

    for line_number, (line, probability) in enumerate(zip(lines, probabilities, strict=True), start=1):
        label = "DELETE" if probability >= THRESHOLD else "KEEP"
        print(f"{line_number:02d} [{label}] {probability:.4f} {line}")

    print(f"Inference time: {inference_time:.4f} seconds")


def predict_line_probabilities(
    lines: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
) -> list[float]:
    text = "".join(f"{LINE_TOKEN}{line}" for line in lines)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.softmax(logits, dim=-1)[:, 1]
    return [float(probability) for probability in probabilities.detach().cpu().tolist()]


if __name__ == "__main__":
    main()
