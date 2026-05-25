from __future__ import annotations

import argparse
from os import environ
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.hf_api import CommitInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "run-001" / "best"
DEFAULT_COMMIT_MESSAGE = "Upload trained FineWeb2 line deleter model"


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
    return resolved_model_dir


def upload_model_to_hub(model_dir: Path, commit_message: str, token: str, repo_id: str) -> CommitInfo:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    return api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )


def main() -> None:
    args = parse_args()
    model_dir = validate_model_dir(args.model_dir)
    token, repo_id = load_huggingface_settings(PROJECT_ROOT / ".env")
    commit_info = upload_model_to_hub(model_dir, args.commit_message, token, repo_id)
    print(f"Uploaded {model_dir} to https://huggingface.co/{repo_id}")
    print(f"Commit: {commit_info.commit_url}")


if __name__ == "__main__":
    main()
