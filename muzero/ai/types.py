from typing import NamedTuple

import numpy as np


class MinibatchArrays(NamedTuple):
    """All arrays are numpy (host) until train_step transfers them to device.

    Shapes:
      phi_stack : (B, q+1, *state_shape)   float32
      actions   : (B, w)                   int32   (actions taken AFTER the current step)
      target_pi : (B, w+1, num_actions)    float32 (policy target at step 0..w)
      target_v  : (B, w+1)                 float32 (value target at step 0..w)
      target_r  : (B, w)                   float32 (reward target for action j -> step j+1)
      mask      : (B, w+1)                 float32 (1 valid, 0 past-episode/terminal)
    """
    phi_stack: np.ndarray
    actions: np.ndarray
    target_pi: np.ndarray
    target_v: np.ndarray
    target_r: np.ndarray
    mask: np.ndarray


class SearchResult(NamedTuple):
    visit_counts: np.ndarray  # (num_actions,) int
    policy: np.ndarray        # (num_actions,) float, sums to 1
    root_value: float
    q_values: np.ndarray      # (num_actions,) per-action Q from the root's perspective
