from __future__ import annotations

from pathlib import Path

import yaml

from muzero.ai.nn.manager import NeuralNetworkManager


def save_checkpoint(nnm: NeuralNetworkManager, path: str | Path,
                    config_snapshot: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nnm.save(path)
    if config_snapshot is not None:
        with open(path.with_suffix(".config.yaml"), "w") as f:
            yaml.safe_dump(config_snapshot, f)


def load_checkpoint(nnm: NeuralNetworkManager, path: str | Path) -> None:
    nnm.load(Path(path))
