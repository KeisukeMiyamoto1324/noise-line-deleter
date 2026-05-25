from __future__ import annotations

import argparse
import csv
import json
import math
import random
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from fineweb2_line_deleter.config import TrainConfig
from fineweb2_line_deleter.data import (
    LineWindow,
    LineWindowDataset,
    collate_line_windows,
    count_labels,
    create_windows,
    dataset_sizes,
    load_line_noise_dataset,
)
from fineweb2_line_deleter.metrics import compute_metrics, prefix_metrics
from fineweb2_line_deleter.model import (
    LineNoiseModel,
    create_tokenizer,
    get_line_token_id,
    save_training_artifacts,
)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="sbintuitions/modernbert-ja-130m")
    parser.add_argument("--dataset-name", type=str, default="MK0727/line-noise-label")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/run-001"))
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-lines-per-window", type=int, default=512)
    parser.add_argument("--line-overlap", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze-until-layer", type=int, default=10)
    parser.add_argument("--loss-log-steps", type=int, default=1)
    args = parser.parse_args()
    if args.loss_log_steps <= 0:
        raise ValueError("--loss-log-steps must be greater than 0.")
    return TrainConfig(**vars(args))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_mps_device() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available.")
    return torch.device("mps")


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = get_mps_device()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = create_tokenizer(config.model_name)
    line_token_id = get_line_token_id(tokenizer)
    dataset_dict = load_line_noise_dataset(
        config.dataset_name,
        config.train_ratio,
        config.valid_ratio,
        config.seed,
    )

    windows_by_split = {
        split: create_windows(
            dataset=dataset,
            tokenizer=tokenizer,
            line_token_id=line_token_id,
            max_length=config.max_length,
            max_lines_per_window=config.max_lines_per_window,
            line_overlap=config.line_overlap,
        )
        for split, dataset in dataset_dict.items()
    }

    model = LineNoiseModel(
        config.model_name,
        line_token_id,
        len(tokenizer),
        config.freeze_until_layer,
    ).to(device)
    clean_count, noise_count = count_labels(windows_by_split["train"])
    class_weights = torch.tensor(
        [
            (clean_count + noise_count) / (2.0 * clean_count),
            (clean_count + noise_count) / (2.0 * noise_count),
        ],
        dtype=torch.float32,
        device=device,
    )

    train_loader = create_loader(windows_by_split["train"], tokenizer.pad_token_id, config, shuffle=True)
    valid_loader = create_loader(windows_by_split["valid"], tokenizer.pad_token_id, config, shuffle=False)
    test_loader = create_loader(windows_by_split["test"], tokenizer.pad_token_id, config, shuffle=False)

    optimizer = AdamW(get_trainable_parameters(model), lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_training_steps = optimizer_steps_per_epoch * config.epochs
    warmup_steps = int(total_training_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_training_steps)

    write_json(config.output_dir / "config.json", config.to_json_dict())
    write_json(config.output_dir / "dataset_sizes.json", dataset_sizes(dataset_dict))
    write_json(config.output_dir / "parameter_summary.json", model.parameter_summary())
    initialize_loss_csv(config.output_dir / "losses.csv")

    best_f1 = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            class_weights,
            device,
            config.gradient_accumulation_steps,
            config.max_grad_norm,
            epoch,
            config.output_dir / "losses.csv",
            config.loss_log_steps,
        )
        valid_metrics = evaluate(model, valid_loader, class_weights, device)
        epoch_metrics = {"epoch": epoch, "train_loss": train_loss, **prefix_metrics(valid_metrics, "valid")}
        history.append(epoch_metrics)
        write_json(config.output_dir / "history.json", history)

        if valid_metrics["f1_noise"] > best_f1:
            best_f1 = valid_metrics["f1_noise"]
            save_training_artifacts(model, tokenizer, config.output_dir / "best")

    test_metrics = evaluate(model, test_loader, class_weights, device)
    last_metrics = {"best_valid_f1_noise": best_f1, **prefix_metrics(test_metrics, "test")}
    write_json(config.output_dir / "metrics.json", last_metrics)
    save_training_artifacts(model, tokenizer, config.output_dir / "last")


def create_loader(
    windows: list[LineWindow],
    pad_token_id: int | None,
    config: TrainConfig,
    shuffle: bool,
) -> DataLoader[dict[str, torch.Tensor | list[list[int]]]]:
    if pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is required.")
    dataset = LineWindowDataset(windows)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=partial(collate_line_windows, pad_token_id=pad_token_id),
    )


def get_trainable_parameters(model: LineNoiseModel) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def train_one_epoch(
    model: LineNoiseModel,
    loader: DataLoader[dict[str, torch.Tensor | list[list[int]]]],
    optimizer: AdamW,
    scheduler: Any,
    class_weights: torch.Tensor,
    device: torch.device,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    epoch: int,
    loss_csv_path: Path,
    loss_log_steps: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    accumulation_loss = 0.0
    accumulation_steps = 0
    optimizer_step = 0
    progress = tqdm(loader, desc=f"epoch {epoch}", leave=False)

    for step, batch in enumerate(progress, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        outputs = model(input_ids, attention_mask, labels, class_weights)
        loss = outputs["loss"] / gradient_accumulation_steps
        loss.backward()
        batch_loss = float(outputs["loss"].detach().cpu().item())
        total_loss += batch_loss
        accumulation_loss += batch_loss
        accumulation_steps += 1

        if step % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            accumulation_average_loss = accumulation_loss / accumulation_steps
            if optimizer_step % loss_log_steps == 0:
                append_optimizer_step_loss_csv(loss_csv_path, epoch, optimizer_step, accumulation_average_loss)
            accumulation_loss = 0.0
            accumulation_steps = 0

    if len(loader) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        accumulation_average_loss = accumulation_loss / accumulation_steps
        if optimizer_step % loss_log_steps == 0:
            append_optimizer_step_loss_csv(loss_csv_path, epoch, optimizer_step, accumulation_average_loss)

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: LineNoiseModel,
    loader: DataLoader[dict[str, torch.Tensor | list[list[int]]]],
    class_weights: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for batch in tqdm(loader, desc="evaluate", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        outputs = model(input_ids, attention_mask, labels, class_weights)
        predictions = torch.argmax(outputs["logits"], dim=-1).detach().cpu().tolist()
        flat_labels = [label for row_labels in labels for label in row_labels]
        all_predictions.extend(int(prediction) for prediction in predictions)
        all_labels.extend(int(label) for label in flat_labels)
        total_loss += float(outputs["loss"].detach().cpu().item())

    metrics = compute_metrics(all_labels, all_predictions)
    metrics["loss"] = total_loss / len(loader)
    return metrics


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def initialize_loss_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "optimizer_step", "train_loss"])


def append_optimizer_step_loss_csv(path: Path, epoch: int, optimizer_step: int, train_loss: float) -> None:
    with path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([epoch, optimizer_step, train_loss])


if __name__ == "__main__":
    main()
