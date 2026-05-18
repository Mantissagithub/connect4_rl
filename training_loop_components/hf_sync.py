from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class HFConfig:
    username: str
    token: str
    collection_name: str
    private: bool = False

    @property
    def repo_id(self) -> str:
        return f"{self.username}/{self.collection_name}"


class HFCheckpointUploader:
    def __init__(self, config: HFConfig):
        self.config = config
        self._initialized = False

    def _ensure_repo(self):
        if self._initialized:
            return

        try:
            from huggingface_hub import create_repo
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is not installed. Install it with `pip install huggingface_hub`."
            ) from exc

        create_repo(
            repo_id=self.config.repo_id,
            token=self.config.token,
            private=self.config.private,
            exist_ok=True,
            repo_type="model",
        )
        self._initialized = True

    def upload_checkpoint(self, checkpoint_path: str, final: bool = False) -> str:
        self._ensure_repo()

        try:
            from huggingface_hub import upload_file
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is not installed. Install it with `pip install huggingface_hub`."
            ) from exc

        filename = os.path.basename(checkpoint_path)
        repo_path = f"checkpoints/{filename}"
        upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=repo_path,
            repo_id=self.config.repo_id,
            token=self.config.token,
            repo_type="model",
            commit_message=f"Add {filename}" if not final else f"Add final checkpoint {filename}",
        )
        return repo_path
