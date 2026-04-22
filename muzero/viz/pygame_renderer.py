"""Game-agnostic pygame shell. Delegates drawing to SimWorld.render_frame.

The only window-management code in the project. Knows nothing about specific
games; BitFall and TicTacToe each implement their own render_frame.
"""
from __future__ import annotations

import os
from typing import Any

from configs._schema import VizConfig


class PygameRenderer:
    def __init__(self, cfg: VizConfig, state_shape: tuple[int, ...]):
        # Delay pygame import so this module stays importable in headless mode.
        # Also: init pygame BEFORE any JAX call to avoid SDL/CUDA context clashes
        # on macOS. Caller is responsible for construction order.
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        self._pygame = pygame
        pygame.init()

        cell = cfg.cell_size_px
        if len(state_shape) == 3:
            h_cells, w_cells, _ = state_shape
        elif len(state_shape) == 2:
            h_cells, w_cells = state_shape
        else:
            h_cells = w_cells = 8  # fallback for vector states

        width = max(3, w_cells) * cell
        height = max(3, h_cells) * cell + 60  # extra room for HUD

        self._screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(cfg.window_title)
        self._clock = pygame.time.Clock()
        self._cfg = cfg
        self._board_rect = pygame.Rect(0, 0, width, height - 60)
        self._hud_rect = pygame.Rect(0, height - 60, width, 60)
        self._font = pygame.font.Font(None, 24)

    def render(self, simworld: Any, state: Any, info: dict | None = None) -> None:
        pygame = self._pygame
        # pump events so the window stays responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self._screen.fill((20, 20, 28))
        board_surface = self._screen.subsurface(self._board_rect)
        simworld.render_frame(board_surface, state)

        pygame.draw.rect(self._screen, (10, 10, 14), self._hud_rect)
        if info:
            text = " | ".join(f"{k}={v}" for k, v in info.items())
            surf = self._font.render(text, True, (230, 230, 230))
            self._screen.blit(surf, (8, self._hud_rect.top + 20))

        pygame.display.flip()
        self._clock.tick(self._cfg.pygame_fps)

    def close(self) -> None:
        self._pygame.quit()
