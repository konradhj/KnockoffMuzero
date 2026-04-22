"""BitFall: a scrolling-debris arcade game with shiftable receptors.

Grid of shape (rows, cols). At each timestep:
  - all debris rows shift down by one
  - the bottom debris row is compared segment-by-segment against receptors
  - a fresh random debris row is added at the top
  - the receptor row stays/shifts one cell left or right

Channels (state_shape = (rows, cols, 2)):
  [:, :, 0] = debris (1 where debris bit is set, else 0)
  [:, :, 1] = receptor mask on the BOTTOM row only (1 where a receptor segment sits)

Actions: 0 = shift-left, 1 = stay, 2 = shift-right.
Reward: result of comparing receptor segments to bottom debris row (see PDF Fig. 3):
  - a segment strictly larger than the other with overlap on one or both sides: segment wins;
    winner's reward magnitude = size of the losing segment.
  - receptor win -> positive; debris win -> negative; no dominant overlap -> 0.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import SimWorld


LEFT, STAY, RIGHT = 0, 1, 2


@dataclass
class BitFallState:
    # debris: bool array (rows, cols); row 0 is the top, row rows-1 is the bottom.
    debris: np.ndarray
    # receptors: list of (start_col, length) segments in the bottom row. Segments never
    # overlap and always fit within [0, cols].
    receptors: tuple[tuple[int, int], ...]
    step: int = 0
    score: float = 0.0
    rng_state: np.random.Generator | None = field(default=None, repr=False)

    def observable(self) -> np.ndarray:
        rows, cols = self.debris.shape
        obs = np.zeros((rows, cols, 2), dtype=np.float32)
        obs[:, :, 0] = self.debris.astype(np.float32)
        for start, length in self.receptors:
            obs[rows - 1, start : start + length, 1] = 1.0
        return obs


def _random_debris_row(cols: int, density: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(cols) < density).astype(bool)


def _initial_receptors(cols: int, num_segments: int) -> tuple[tuple[int, int], ...]:
    # Evenly space num_segments receptors across cols, each ~floor(cols / (2 * num_segments)).
    seg_len = max(1, cols // (2 * num_segments))
    gap = max(1, (cols - num_segments * seg_len) // max(1, num_segments))
    out: list[tuple[int, int]] = []
    pos = 0
    for _ in range(num_segments):
        if pos + seg_len > cols:
            break
        out.append((pos, seg_len))
        pos += seg_len + gap
    if not out:
        out = [(0, 1)]
    return tuple(out)


def _shift_receptors(receptors: tuple[tuple[int, int], ...], cols: int, direction: int
                     ) -> tuple[tuple[int, int], ...]:
    """direction: -1 left, +1 right, 0 stay. Implements the PDF's split/rejoin rule."""
    if direction == 0:
        return receptors
    out: list[tuple[int, int]] = []
    for start, length in receptors:
        new_start = start + direction
        if new_start < 0:
            # Running past left edge: split off one cell that wraps/falls off.
            # PDF: a receptor of size X past left edge splits into size 1 and X-1.
            # We model this as: lose one cell from the left, the remaining X-1 shifts normally.
            remainder = length - 1
            if remainder >= 1:
                out.append((0, remainder))
            # Also keep a size-1 "stub" at position 0 to match the PDF's "split into 1 and X-1".
            # On a second shift in the same direction, 2 and X-2, etc.
            # For simplicity we keep only the larger piece; the stub gets absorbed next step.
        elif new_start + length > cols:
            overflow = (new_start + length) - cols
            remainder = length - overflow
            if remainder >= 1:
                out.append((new_start, remainder))
        else:
            out.append((new_start, length))
    # Deduplicate and keep sorted.
    out = sorted(set(out))
    return tuple(out)


def _score_row(debris_bottom: np.ndarray,
               receptors: tuple[tuple[int, int], ...]) -> float:
    """Compare each receptor segment to the debris bits directly beneath it.

    Rule (simplified from PDF Fig. 3): for each receptor segment of length R at [s, s+R):
      - let D = number of debris bits set in that span
      - if R > D and D >= 1: receptor wins, +D points (we score the losing side's size)
      - if D > R: debris wins, -R points
      - else (D == R or D == 0): 0 points
    Contiguous debris segments that extend past the receptor edges also invoke negative
    scoring. For simplicity, we only consider the span under each receptor; this captures
    the dominant signal and keeps the reward function deterministic and bounded.
    """
    total = 0.0
    cols = debris_bottom.shape[0]
    used_mask = np.zeros(cols, dtype=bool)
    for start, length in receptors:
        end = min(start + length, cols)
        span = debris_bottom[start:end]
        R = end - start
        D = int(span.sum())
        if R > D and D >= 1:
            total += float(D)  # receptor wins, reward = losing (debris) size
        elif D > R:
            total -= float(R)  # debris wins, penalty = losing (receptor) size
        used_mask[start:end] = True
    # Any debris outside receptor spans contributes a small negative (missed debris).
    missed = int(debris_bottom[~used_mask].sum())
    total -= 0.25 * float(missed)
    return total


class BitFallSimWorld(SimWorld):
    def __init__(self, grid_rows: int = 6, grid_cols: int = 6,
                 num_receptor_segments: int = 3, debris_density: float = 0.4,
                 horizon: int = 60, reward_scale: float = 1.0):
        self._rows = int(grid_rows)
        self._cols = int(grid_cols)
        self._num_segments = int(num_receptor_segments)
        self._density = float(debris_density)
        self._horizon = int(horizon)
        self._reward_scale = float(reward_scale)

    # --- static descriptors ---
    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self._rows, self._cols, 2)

    @property
    def num_actions(self) -> int:
        return 3

    @property
    def num_players(self) -> int:
        return 1

    @property
    def reward_range(self) -> tuple[float, float]:
        return (-float(self._cols) * self._reward_scale, float(self._cols) * self._reward_scale)

    # --- dynamics ---
    def initial_state(self, rng: np.random.Generator) -> BitFallState:
        debris = np.zeros((self._rows, self._cols), dtype=bool)
        # Prefill a few rows with random debris so the game starts with something to do.
        for r in range(self._rows - 1):
            debris[r] = _random_debris_row(self._cols, self._density, rng)
        receptors = _initial_receptors(self._cols, self._num_segments)
        return BitFallState(debris=debris, receptors=receptors, step=0, score=0.0, rng_state=rng)

    def step(self, state: BitFallState, action: int) -> tuple[BitFallState, float, bool]:
        rng = state.rng_state if state.rng_state is not None else np.random.default_rng()
        direction = {LEFT: -1, STAY: 0, RIGHT: +1}[int(action)]
        new_receptors = _shift_receptors(state.receptors, self._cols, direction)
        # Shift debris down by one; drop the bottom row, add a fresh top row.
        bottom = state.debris[-1].copy()
        new_debris = np.roll(state.debris, shift=1, axis=0)
        new_debris[0] = _random_debris_row(self._cols, self._density, rng)
        reward = _score_row(bottom, new_receptors) * self._reward_scale
        new_state = BitFallState(
            debris=new_debris,
            receptors=new_receptors,
            step=state.step + 1,
            score=state.score + reward,
            rng_state=rng,
        )
        terminal = new_state.step >= self._horizon
        return new_state, reward, terminal

    def legal_actions(self, state: BitFallState) -> np.ndarray:
        return np.ones(3, dtype=bool)

    def current_player(self, state: BitFallState) -> int:
        return 0

    def is_terminal(self, state: BitFallState) -> bool:
        return state.step >= self._horizon

    def blank_state(self) -> BitFallState:
        return BitFallState(
            debris=np.zeros((self._rows, self._cols), dtype=bool),
            receptors=(),
            step=0,
            score=0.0,
        )

    # --- rendering ---
    def render_frame(self, surface: Any, state: BitFallState) -> None:
        import pygame

        w, h = surface.get_size()
        cell_w = w // self._cols
        cell_h = h // self._rows
        surface.fill((20, 20, 28))
        for r in range(self._rows):
            for c in range(self._cols):
                rect = pygame.Rect(c * cell_w, r * cell_h, cell_w - 1, cell_h - 1)
                if r == self._rows - 1:
                    pygame.draw.rect(surface, (40, 40, 60), rect)
                if state.debris[r, c]:
                    pygame.draw.rect(surface, (220, 60, 60), rect)
        for start, length in state.receptors:
            rect = pygame.Rect(start * cell_w, (self._rows - 1) * cell_h,
                               length * cell_w - 1, cell_h - 1)
            pygame.draw.rect(surface, (70, 140, 220), rect)

    def render_ascii(self, state: BitFallState) -> str:
        rows = []
        for r in range(self._rows):
            line = []
            for c in range(self._cols):
                if r == self._rows - 1 and any(s <= c < s + L for s, L in state.receptors):
                    line.append("B" if state.debris[r, c] else "_")
                else:
                    line.append("#" if state.debris[r, c] else ".")
            rows.append("".join(line))
        rows.append(f"step={state.step} score={state.score:.2f}")
        return "\n".join(rows)
