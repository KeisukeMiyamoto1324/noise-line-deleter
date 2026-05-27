from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutput

from corpus_refiner_jp import LINE_TOKEN


class LineKeepModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        line_token_id: int,
        tokenizer_length: int,
        freeze_until_layer: int,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.resize_token_embeddings(tokenizer_length)
        self.dropout = nn.Dropout(float(self.encoder.config.classifier_dropout))
        self.classifier = nn.Linear(int(self.encoder.config.hidden_size), 2)
        self.line_token_id = line_token_id
        self.freeze_until_layer = freeze_until_layer
        self.freeze_encoder_layers(freeze_until_layer)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: list[list[int]] | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        line_hidden_states = self.extract_line_hidden_states(encoder_outputs, input_ids, labels)
        logits = self.classifier(self.dropout(line_hidden_states))
        output: dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            flat_labels = torch.tensor(
                [label for row_labels in labels for label in row_labels],
                dtype=torch.long,
                device=logits.device,
            )
            loss_function = nn.CrossEntropyLoss(weight=class_weights)
            output["loss"] = loss_function(logits, flat_labels)

        return output

    def extract_line_hidden_states(
        self,
        encoder_outputs: BaseModelOutput,
        input_ids: torch.Tensor,
        labels: list[list[int]] | None,
    ) -> torch.Tensor:
        hidden_states = encoder_outputs.last_hidden_state
        line_mask = input_ids == self.line_token_id
        if labels is not None:
            expected_count = sum(len(row_labels) for row_labels in labels)
            actual_count = int(line_mask.sum().item())
            if actual_count != expected_count:
                raise ValueError(f"Expected {expected_count} line tokens, found {actual_count}.")
        return hidden_states[line_mask]

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.config.save_pretrained(output_dir)
        torch.save(self.state_dict(), output_dir / "model.pt")

    def freeze_encoder_layers(self, freeze_until_layer: int) -> None:
        layer_count = len(self.encoder.layers)
        if freeze_until_layer < 0 or freeze_until_layer > layer_count:
            raise ValueError(f"freeze_until_layer must be between 0 and {layer_count}.")

        for parameter in self.encoder.embeddings.parameters():
            parameter.requires_grad = False

        for layer_index, layer in enumerate(self.encoder.layers):
            requires_grad = layer_index >= freeze_until_layer
            for parameter in layer.parameters():
                parameter.requires_grad = requires_grad

    def parameter_summary(self) -> dict[str, Any]:
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        frozen_parameters = total_parameters - trainable_parameters
        return {
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "frozen_parameters": frozen_parameters,
            "trainable_ratio": trainable_parameters / total_parameters,
            "freeze_until_layer": self.freeze_until_layer,
        }


def create_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": [LINE_TOKEN]})
    return tokenizer


def get_line_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    line_token_id = tokenizer.convert_tokens_to_ids(LINE_TOKEN)
    if not isinstance(line_token_id, int):
        raise ValueError("Line token id is not an integer.")
    return line_token_id


def save_training_artifacts(
    model: LineKeepModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir)
    tokenizer.save_pretrained(output_dir)
    AutoConfig.from_pretrained(output_dir).save_pretrained(output_dir)
