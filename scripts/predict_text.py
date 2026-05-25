from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutput


MODEL_REPO_ID = "MK0727/noise-line-remover-jp"
BASE_MODEL_NAME = "sbintuitions/modernbert-ja-130m"
LINE_TOKEN = "<line>"
MAX_LENGTH = 4096
LINE_OVERLAP = 4
THRESHOLD = 0.5
TEXT = """

"""


@dataclass(frozen=True)
class InferenceWindow:
    input_ids: list[int]
    attention_mask: list[int]
    line_indices: list[int]


class LineNoiseModel(nn.Module):
    def __init__(self, model_name: str, line_token_id: int, tokenizer_length: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.resize_token_embeddings(tokenizer_length)
        self.dropout = nn.Dropout(float(self.encoder.config.classifier_dropout))
        self.classifier = nn.Linear(int(self.encoder.config.hidden_size), 2)
        self.line_token_id = line_token_id

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        line_hidden_states = self.extract_line_hidden_states(encoder_outputs, input_ids)
        return self.classifier(self.dropout(line_hidden_states))

    def extract_line_hidden_states(self, encoder_outputs: BaseModelOutput, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = encoder_outputs.last_hidden_state
        line_mask = input_ids == self.line_token_id
        return hidden_states[line_mask]


def main() -> None:
    device = torch.device("mps")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
    line_token_id = get_line_token_id(tokenizer)
    model_path = Path(hf_hub_download(repo_id=MODEL_REPO_ID, filename="model.pt"))
    model = LineNoiseModel(BASE_MODEL_NAME, line_token_id, len(tokenizer))
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
            logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
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


def wrap_with_special_tokens(tokenizer: PreTrainedTokenizerBase, token_ids: list[int]) -> list[int]:
    special_token_ids = tokenizer.encode("", add_special_tokens=True)
    return special_token_ids[:1] + token_ids + special_token_ids[1:]


def get_line_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    line_token_id = tokenizer.convert_tokens_to_ids(LINE_TOKEN)
    if not isinstance(line_token_id, int):
        raise ValueError("Line token id is not an integer.")
    return line_token_id


if __name__ == "__main__":
    main()
