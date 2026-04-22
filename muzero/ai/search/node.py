from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class UMCTSEdge:
    action: int
    prior: float
    reward: float = 0.0          # r from NN_d at expansion time
    visit_count: int = 0
    total_value: float = 0.0     # W; Q = W / max(1, N)
    child: "UMCTSNode | None" = None

    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


@dataclass
class UMCTSNode:
    sigma: np.ndarray                                   # abstract state (hidden_dim,)
    is_expanded: bool = False
    edges: list[UMCTSEdge] = field(default_factory=list)
    parent_edge: UMCTSEdge | None = None
    to_play: int = 0                                    # player to move (0 in 1-player)
    legal_mask: np.ndarray | None = None                # only set at the root; elsewhere None

    def total_visit_count(self) -> int:
        return sum(e.visit_count for e in self.edges)
