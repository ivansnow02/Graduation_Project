"""Upload a local model directory to the Hugging Face Hub or ModelScope.

By default this targets the final adapter/model artifacts in
`outputs/dpo500softmargin` and skips intermediate checkpoint directories.

Examples:
  python scripts/upload_to_hub.py \
    --repo_id your-username/dpo500softmargin

  python scripts/upload_to_hub.py \
    --repo_id your-username/dpo500softmargin \
    --include_checkpoints

  python scripts/upload_to_hub.py \
    --platform modelscope \
    --repo_id your-username/dpo500softmargin
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

try:
    from modelscope.hub.api import HubApi as ModelScopeHubApi
except ImportError:  # pragma: no cover - optional dependency
    ModelScopeHubApi = None


DEFAULT_LOCAL_PATH = Path("outputs/dpo500softmargin")


def _copy_filtered_tree(src: Path, dst: Path, include_checkpoints: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if not include_checkpoints and item.name.startswith("checkpoint-"):
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _ensure_modelscope_configuration(model_dir: Path) -> None:
    config_path = model_dir / "configuration.json"
    if config_path.exists():
        return

    config = {"framework": "Pytorch", "task": "other"}
    config_path.write_text(json.dumps(config, ensure_ascii=True, indent=2), encoding="utf-8")


def upload_directory(
    local_path: Path,
    repo_id: str,
    *,
    platform: str,
    repo_type: str,
    token: str | None,
    private: bool,
    endpoint: str | None,
    include_checkpoints: bool,
    commit_message: str,
) -> None:
    if not local_path.exists():
        raise FileNotFoundError(f"Local path does not exist: {local_path}")

    if platform == "huggingface":
        if endpoint and "hf-mirror.com" in endpoint:
            raise ValueError(
                "hf-mirror.com is a download mirror, not a reliable write endpoint for Hub uploads. "
                "Omit --endpoint for uploads, or set it to https://huggingface.co."
            )

        api = HfApi(endpoint=endpoint)
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            private=private,
            exist_ok=True,
        )

        ignore_patterns = [".git", ".venv", "__pycache__", "*.pyc"]
        if not include_checkpoints:
            ignore_patterns.extend(["checkpoint-*", "checkpoint-*/*"])

        print(f"Uploading {local_path} -> {repo_id} ({repo_type})")
        if endpoint:
            print(f"Using endpoint: {endpoint}")
        if not include_checkpoints:
            print("Skipping checkpoint-* directories")

        api.upload_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message=commit_message,
            ignore_patterns=ignore_patterns,
        )

    elif platform == "modelscope":
        if ModelScopeHubApi is None:
            raise ImportError(
                "modelscope is not installed. Install it with `pip install modelscope` before uploading to ModelScope."
            )
        if not token:
            raise ValueError(
                "ModelScope uploads require a token. Pass --token or set MODELSCOPE_API_TOKEN."
            )

        upload_source = local_path
        with tempfile.TemporaryDirectory(prefix="modelscope-upload-") as tmp_dir:
            staged = Path(tmp_dir) / local_path.name
            if include_checkpoints:
                shutil.copytree(local_path, staged)
            else:
                _copy_filtered_tree(local_path, staged, include_checkpoints=False)
            _ensure_modelscope_configuration(staged)
            upload_source = staged

            print(f"Uploading {local_path} -> {repo_id} (modelscope)")
            if not include_checkpoints:
                print("Skipping checkpoint-* directories")

            api = ModelScopeHubApi()
            api.login(token)
            if hasattr(api, "upload_folder"):
                api.upload_folder(
                    repo_id=repo_id,
                    folder_path=str(upload_source),
                    commit_message=commit_message,
                )
            else:
                api.push_model(model_id=repo_id, model_dir=str(upload_source))

    else:
        raise ValueError(f"Unsupported platform: {platform}")

    print("Upload complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a local folder to Hugging Face Hub")
    parser.add_argument(
        "--local_path",
        type=Path,
        default=DEFAULT_LOCAL_PATH,
        help="Path to the local model directory",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="Target model hub platform",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Target repo id, for example your-username/dpo500softmargin",
    )
    parser.add_argument(
        "--repo_type",
        type=str,
        default="model",
        choices=["model", "dataset", "space"],
        help="Hub repository type",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("MODELSCOPE_API_TOKEN"),
        help="Access token. Falls back to HF_TOKEN, HUGGINGFACE_HUB_TOKEN, or MODELSCOPE_API_TOKEN.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or keep the repo private",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Hub endpoint for uploads. Leave unset to use the default Hugging Face endpoint.",
    )
    parser.add_argument(
        "--include_checkpoints",
        action="store_true",
        help="Upload checkpoint-* directories too",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload model artifacts",
        help="Commit message for the upload",
    )

    args = parser.parse_args()

    upload_directory(
        local_path=args.local_path,
        repo_id=args.repo_id,
        platform=args.platform,
        repo_type=args.repo_type,
        token=args.token,
        private=args.private,
        endpoint=args.endpoint,
        include_checkpoints=args.include_checkpoints,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
modelscope upload ivansnow02/qwen3-14b-sid-margin outputs/dpo500softmargin
