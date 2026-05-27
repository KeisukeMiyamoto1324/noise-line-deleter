from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from corpus_refiner_jp import LINE_TOKEN
from corpus_refiner_jp.data import build_line_labels, load_line_keep_dataset


MODEL_REPO_ID = "MK0727/corpus-refiner-jp"
DATASET_NAME = "MK0727/noise-line-label-jp"
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
SEED = 42
SAMPLE_COUNT = 100
THRESHOLD = 0.5


def main() -> None:
    samples_dir = Path("samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    device = get_mps_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
    model = AutoModel.from_pretrained(MODEL_REPO_ID, trust_remote_code=True)
    model.to(device)
    model.eval()
    dataset_dict = load_line_keep_dataset(
        DATASET_NAME,
        TRAIN_RATIO,
        VALID_RATIO,
        SEED,
    )
    dataset = dataset_dict["test"].select(range(SAMPLE_COUNT))

    samples = [
        create_sample_payload(
            row=row,
            sample_index=index,
            tokenizer=tokenizer,
            model=model,
            device=device,
            samples_dir=samples_dir,
        )
        for index, row in enumerate(dataset, start=1)
    ]
    payload = {
        "metadata": {
            "dataset_name": DATASET_NAME,
            "model_repo": MODEL_REPO_ID,
            "split": "test",
            "sample_count": len(samples),
            "threshold": THRESHOLD,
        },
        "samples": samples,
    }
    (samples_dir / "predictions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def get_mps_device() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available.")
    return torch.device("mps")


def create_sample_payload(
    row: dict[str, Any],
    sample_index: int,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
    samples_dir: Path,
) -> dict[str, Any]:
    document_id = str(row["id"])
    text = str(row["text"])
    lines, labels = build_line_labels(text, list(row["lines_to_keep"]))
    text_file = f"sample_{sample_index:04d}.txt"
    (samples_dir / text_file).write_text(text, encoding="utf-8")
    probabilities = predict_keep_probabilities(
        text=text,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    line_payloads = [
        {
            "number": line_index + 1,
            "text": line,
            "actual_keep": labels[line_index] == 1,
            "predicted_keep": probabilities[line_index] >= THRESHOLD,
            "keep_probability": probabilities[line_index],
        }
        for line_index, line in enumerate(lines)
    ]
    return {
        "sample_index": sample_index,
        "document_id": document_id,
        "text_file": text_file,
        "line_count": len(lines),
        "actual_keep_count": sum(labels),
        "predicted_keep_count": sum(1 for probability in probabilities if probability >= THRESHOLD),
        "lines": line_payloads,
    }


def predict_keep_probabilities(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
) -> list[float]:
    lines = text.split("\n")
    marked_text = "".join(f"{LINE_TOKEN}{line}" for line in lines)
    inputs = tokenizer(marked_text, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()

    if len(probabilities) != len(lines):
        raise ValueError(f"Expected {len(lines)} predictions, found {len(probabilities)}.")

    return [float(probability) for probability in probabilities]


if __name__ == "__main__":
    main()
