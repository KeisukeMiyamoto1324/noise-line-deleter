from __future__ import annotations

from dataclasses import dataclass
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class LineWindow:
    document_id: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


class LineWindowDataset(TorchDataset[LineWindow]):
    def __init__(self, windows: list[LineWindow]) -> None:
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> LineWindow:
        return self.windows[index]


def build_line_labels(text: str, lines_to_keep: list[int]) -> tuple[list[str], list[int]]:
    lines = text.split("\n")
    keep_numbers = set(lines_to_keep)
    labels = [1 if line_number in keep_numbers else 0 for line_number in range(1, len(lines) + 1)]
    return lines, labels


def split_dataset(dataset: Dataset, train_ratio: float, valid_ratio: float, seed: int) -> DatasetDict:
    train_test = dataset.train_test_split(test_size=1.0 - train_ratio, seed=seed)
    valid_test_ratio = valid_ratio / (1.0 - train_ratio)
    valid_test = train_test["test"].train_test_split(test_size=1.0 - valid_test_ratio, seed=seed)
    return DatasetDict(
        {
            "train": train_test["train"],
            "valid": valid_test["train"],
            "test": valid_test["test"],
        }
    )


def load_line_keep_dataset(dataset_name: str, train_ratio: float, valid_ratio: float, seed: int) -> DatasetDict:
    dataset = load_dataset(dataset_name, split="train")
    return split_dataset(dataset, train_ratio, valid_ratio, seed)


def create_windows(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    line_token_id: int,
    max_length: int,
    max_lines_per_window: int,
    line_overlap: int,
) -> list[LineWindow]:
    windows: list[LineWindow] = []
    for row in dataset:
        windows.extend(
            create_document_windows(
                document_id=str(row["id"]),
                text=str(row["text"]),
                lines_to_keep=list(row["lines_to_keep"]),
                tokenizer=tokenizer,
                line_token_id=line_token_id,
                max_length=max_length,
                max_lines_per_window=max_lines_per_window,
                line_overlap=line_overlap,
            )
        )
    return windows


def create_document_windows(
    document_id: str,
    text: str,
    lines_to_keep: list[int],
    tokenizer: PreTrainedTokenizerBase,
    line_token_id: int,
    max_length: int,
    max_lines_per_window: int,
    line_overlap: int,
) -> list[LineWindow]:
    lines, labels = build_line_labels(text, lines_to_keep)
    encoded_lines = [encode_line(line, tokenizer, line_token_id, max_length) for line in lines]
    windows: list[LineWindow] = []
    start_index = 0

    while start_index < len(encoded_lines):
        current_ids: list[int] = []
        current_labels: list[int] = []
        current_index = start_index

        while current_index < len(encoded_lines) and len(current_labels) < max_lines_per_window:
            candidate_ids = current_ids + encoded_lines[current_index]
            prepared_length = len(wrap_with_special_tokens(tokenizer, candidate_ids))
            if current_labels and prepared_length > max_length:
                break
            if prepared_length > max_length:
                current_ids = trim_single_line(encoded_lines[current_index], tokenizer, max_length)
                current_labels = [labels[current_index]]
                current_index += 1
                break
            current_ids = candidate_ids
            current_labels.append(labels[current_index])
            current_index += 1

        input_ids = wrap_with_special_tokens(tokenizer, current_ids)
        windows.append(
            LineWindow(
                document_id=document_id,
                input_ids=input_ids,
                attention_mask=[1] * len(input_ids),
                labels=current_labels,
            )
        )

        if current_index >= len(encoded_lines):
            start_index = current_index
        else:
            retained_lines = min(line_overlap, len(current_labels) - 1)
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


def collate_line_windows(batch: list[LineWindow], pad_token_id: int) -> dict[str, torch.Tensor | list[list[int]]]:
    max_input_length = max(len(item.input_ids) for item in batch)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []

    for item in batch:
        pad_length = max_input_length - len(item.input_ids)
        input_ids.append(item.input_ids + [pad_token_id] * pad_length)
        attention_mask.append(item.attention_mask + [0] * pad_length)
        labels.append(item.labels)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": labels,
    }


def count_labels(windows: list[LineWindow]) -> tuple[int, int]:
    delete_count = 0
    keep_count = 0
    for window in windows:
        for label in window.labels:
            if label == 1:
                keep_count += 1
            else:
                delete_count += 1
    return delete_count, keep_count


def dataset_sizes(dataset_dict: DatasetDict) -> dict[str, int]:
    return {name: len(dataset) for name, dataset in dataset_dict.items()}
