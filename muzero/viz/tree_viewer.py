"""Tree viewer for demo mode. Renders a u-MCTS root + its children as a radial
diagram using matplotlib. Saved to disk as a single png (or an animated GIF
if called repeatedly)."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from muzero.ai.search.node import UMCTSNode


def draw_tree(root: UMCTSNode, out_path: str | Path, max_depth: int = 2) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.axis("off")

    def walk(node: UMCTSNode, cx: float, cy: float, radius: float, depth: int):
        circle = plt.Circle((cx, cy), 0.04 * max(0.6, 1.2 - 0.3 * depth),
                            color="#bfdbfe", ec="#1d4ed8", lw=1.2)
        ax.add_patch(circle)
        ax.text(cx, cy, str(sum(e.visit_count for e in node.edges)),
                ha="center", va="center", fontsize=8)
        if depth >= max_depth or not node.edges:
            return
        edges = [e for e in node.edges if e.child is not None]
        if not edges:
            return
        arc = math.tau / max(1, len(edges))
        for i, e in enumerate(edges):
            angle = i * arc + (0.1 * depth)
            ccx = cx + radius * math.cos(angle)
            ccy = cy + radius * math.sin(angle)
            ax.plot([cx, ccx], [cy, ccy], color="#64748b", lw=0.7)
            mid_x = (cx + ccx) / 2
            mid_y = (cy + ccy) / 2
            label = f"a={e.action}\nN={e.visit_count}\nQ={e.q_value():.2f}"
            ax.text(mid_x, mid_y, label, fontsize=6,
                    ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.8))
            walk(e.child, ccx, ccy, radius * 0.55, depth + 1)

    walk(root, 0.0, 0.0, 0.45, 0)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
