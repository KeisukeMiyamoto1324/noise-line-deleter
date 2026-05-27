from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerBase
from transformers import AutoTokenizer

from fineweb2_line_keeper.data import build_line_labels, load_line_keep_dataset, wrap_with_special_tokens
from fineweb2_line_keeper.model import LineKeepModel, get_line_token_id


MODEL_REPO_ID = "MK0727/noise-line-keeper-jp"


@dataclass(frozen=True)
class InferenceWindow:
    input_ids: list[int]
    attention_mask: list[int]
    line_indices: list[int]


def main() -> None:
    run_dir = Path("outputs/run-001")
    config = read_json(run_dir / "config.json")
    samples_dir = Path("samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    device = get_mps_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
    line_token_id = get_line_token_id(tokenizer)
    model_path = Path(hf_hub_download(repo_id=MODEL_REPO_ID, filename="model.pt"))
    model = load_model(config, model_path, line_token_id, len(tokenizer), device)
    dataset_dict = load_line_keep_dataset(
        str(config["dataset_name"]),
        float(config["train_ratio"]),
        float(config["valid_ratio"]),
        int(config["seed"]),
    )
    dataset = dataset_dict["test"].select(range(100))

    samples = [
        create_sample_payload(
            row=row,
            sample_index=index,
            tokenizer=tokenizer,
            line_token_id=line_token_id,
            model=model,
            device=device,
            config=config,
            samples_dir=samples_dir,
        )
        for index, row in enumerate(dataset, start=1)
    ]
    payload = {
        "metadata": {
            "dataset_name": config["dataset_name"],
            "model_repo": MODEL_REPO_ID,
            "split": "test",
            "sample_count": len(samples),
            "threshold": 0.5,
        },
        "samples": samples,
    }
    (samples_dir / "predictions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_mps_device() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available.")
    return torch.device("mps")


def load_model(
    config: dict[str, Any],
    model_path: Path,
    line_token_id: int,
    tokenizer_length: int,
    device: torch.device,
) -> LineKeepModel:
    model = LineKeepModel(
        str(config["model_name"]),
        line_token_id,
        tokenizer_length,
        int(config["freeze_until_layer"]),
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def create_sample_payload(
    row: dict[str, Any],
    sample_index: int,
    tokenizer: PreTrainedTokenizerBase,
    line_token_id: int,
    model: LineKeepModel,
    device: torch.device,
    config: dict[str, Any],
    samples_dir: Path,
) -> dict[str, Any]:
    document_id = str(row["id"])
    text = str(row["text"])
    lines, labels = build_line_labels(text, list(row["lines_to_keep"]))
    text_file = f"sample_{sample_index:04d}.txt"
    (samples_dir / text_file).write_text(text, encoding="utf-8")
    probabilities = predict_keep_probabilities(
        text=text,
        lines_to_keep=list(row["lines_to_keep"]),
        tokenizer=tokenizer,
        line_token_id=line_token_id,
        model=model,
        device=device,
        max_length=int(config["max_length"]),
        max_lines_per_window=int(config["max_lines_per_window"]),
        line_overlap=int(config["line_overlap"]),
    )
    line_payloads = [
        {
            "number": line_index + 1,
            "text": line,
            "actual_keep": labels[line_index] == 1,
            "predicted_keep": probabilities[line_index] >= 0.5,
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
        "predicted_keep_count": sum(1 for probability in probabilities if probability >= 0.5),
        "lines": line_payloads,
    }


def predict_keep_probabilities(
    text: str,
    lines_to_keep: list[int],
    tokenizer: PreTrainedTokenizerBase,
    line_token_id: int,
    model: LineKeepModel,
    device: torch.device,
    max_length: int,
    max_lines_per_window: int,
    line_overlap: int,
) -> list[float]:
    lines, _ = build_line_labels(text, lines_to_keep)
    windows = create_indexed_windows(
        text=text,
        lines_to_keep=lines_to_keep,
        tokenizer=tokenizer,
        line_token_id=line_token_id,
        max_length=max_length,
        max_lines_per_window=max_lines_per_window,
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
    text: str,
    lines_to_keep: list[int],
    tokenizer: PreTrainedTokenizerBase,
    line_token_id: int,
    max_length: int,
    max_lines_per_window: int,
    line_overlap: int,
) -> list[InferenceWindow]:
    lines, _ = build_line_labels(text, lines_to_keep)
    encoded_lines = [encode_line(line, tokenizer, line_token_id, max_length) for line in lines]
    windows: list[InferenceWindow] = []
    start_index = 0

    while start_index < len(encoded_lines):
        current_ids: list[int] = []
        line_indices: list[int] = []
        current_index = start_index

        while current_index < len(encoded_lines) and len(line_indices) < max_lines_per_window:
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


def encode_line(
    line: str,
    tokenizer: PreTrainedTokenizerBase,
    line_token_id: int,
    max_length: int,
) -> list[int]:
    token_ids = tokenizer.encode(line, add_special_tokens=False)
    max_line_tokens = max_length - len(wrap_with_special_tokens(tokenizer, [line_token_id]))
    return [line_token_id] + token_ids[:max_line_tokens]


def trim_single_line(
    encoded_line: list[int],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> list[int]:
    special_token_count = len(wrap_with_special_tokens(tokenizer, []))
    return encoded_line[: max_length - special_token_count]


if __name__ == "__main__":
    main()
