from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    dataset_name: str
    output_dir: Path
    max_length: int
    max_lines_per_window: int
    line_overlap: int
    train_ratio: float
    valid_ratio: float
    seed: int
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    num_workers: int
    freeze_until_layer: int

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data
