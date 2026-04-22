from configs._schema import GameConfig

from .base import GameState, SimWorld
from .bitfall import BitFallSimWorld
from .tictactoe import TicTacToeSimWorld

_REGISTRY: dict[str, type[SimWorld]] = {
    "bitfall": BitFallSimWorld,
    "tictactoe": TicTacToeSimWorld,
}


def build_simworld(cfg: GameConfig) -> SimWorld:
    if cfg.name not in _REGISTRY:
        raise ValueError(f"Unknown game: {cfg.name}. Known: {list(_REGISTRY)}")
    return _REGISTRY[cfg.name](**cfg.params)


__all__ = ["GameState", "SimWorld", "BitFallSimWorld", "TicTacToeSimWorld", "build_simworld"]
