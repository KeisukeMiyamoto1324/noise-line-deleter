from __future__ import annotations

import argparse
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.hf_api import CommitInfo
from transformers import AutoConfig, AutoTokenizer
from transformers.models.modernbert.modeling_modernbert import ModernBertModel, ModernBertPreTrainedModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "run-001" / "best"
DEFAULT_COMMIT_MESSAGE = "Upload trained FineWeb2 line deleter model"
LINE_TOKEN = "<line>"
REMOTE_MODEL_CODE = '''from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.modernbert.modeling_modernbert import ModernBertModel, ModernBertPreTrainedModel


class LineNoiseModel(ModernBertPreTrainedModel):
    all_tied_weights_keys: dict[str, list[str]] = {}

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.encoder = ModernBertModel(config)
        self.dropout = nn.Dropout(float(config.classifier_dropout))
        self.classifier = nn.Linear(int(config.hidden_size), int(config.num_labels))
        self.line_token_id = int(config.line_token_id)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> TokenClassifierOutput:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        line_hidden_states = outputs.last_hidden_state[input_ids == self.line_token_id]
        logits = self.classifier(self.dropout(line_hidden_states))
        return TokenClassifierOutput(logits=logits)
'''


class LineNoiseModel(ModernBertPreTrainedModel):
    all_tied_weights_keys: dict[str, list[str]] = {}

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.encoder = ModernBertModel(config)
        self.dropout = torch.nn.Dropout(float(config.classifier_dropout))
        self.classifier = torch.nn.Linear(int(config.hidden_size), int(config.num_labels))
        self.line_token_id = int(config.line_token_id)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        line_hidden_states = outputs.last_hidden_state[input_ids == self.line_token_id]
        return self.classifier(self.dropout(line_hidden_states))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--commit-message", type=str, default=DEFAULT_COMMIT_MESSAGE)
    return parser.parse_args()


def load_huggingface_settings(env_path: Path) -> tuple[str, str]:
    if not env_path.is_file():
        raise FileNotFoundError(f".env file does not exist: {env_path}")
    load_dotenv(env_path, override=True)

    token = get_required_env_value("HF_TOKEN")
    repo_id = get_required_env_value("HF_MODEL_REPO_ID")
    return token, repo_id


def get_required_env_value(name: str) -> str:
    value = environ.get(name)
    if value is None or value == "":
        raise ValueError(f"{name} is required in .env.")
    return value


def validate_model_dir(model_dir: Path) -> Path:
    resolved_model_dir = model_dir.expanduser().resolve()
    if not resolved_model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {resolved_model_dir}")
    if not (resolved_model_dir / "model.pt").is_file():
        raise FileNotFoundError(f"model.pt does not exist: {resolved_model_dir / 'model.pt'}")
    return resolved_model_dir


def upload_model_to_hub(model_dir: Path, commit_message: str, token: str, repo_id: str) -> CommitInfo:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    with TemporaryDirectory() as temporary_dir:
        upload_dir = convert_model_dir(model_dir, Path(temporary_dir))
        return api.upload_folder(
            folder_path=upload_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )


def convert_model_dir(model_dir: Path, output_dir: Path) -> Path:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    line_token_id = int(tokenizer.convert_tokens_to_ids(LINE_TOKEN))
    config = AutoConfig.from_pretrained(model_dir)
    config.architectures = ["LineNoiseModel"]
    config.auto_map = {"AutoModel": "modeling_line_noise.LineNoiseModel"}
    config.id2label = {0: "KEEP", 1: "DELETE"}
    config.label2id = {"KEEP": 0, "DELETE": 1}
    config.line_token_id = line_token_id
    config.num_labels = 2

    model = LineNoiseModel(config)
    state_dict = torch.load(model_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "modeling_line_noise.py").write_text(REMOTE_MODEL_CODE, encoding="utf-8")
    return output_dir


def main() -> None:
    args = parse_args()
    model_dir = validate_model_dir(args.model_dir)
    token, repo_id = load_huggingface_settings(PROJECT_ROOT / ".env")
    commit_info = upload_model_to_hub(model_dir, args.commit_message, token, repo_id)
    print(f"Uploaded {model_dir} to https://huggingface.co/{repo_id}")
    print(f"Commit: {commit_info.commit_url}")


if __name__ == "__main__":
    main()
