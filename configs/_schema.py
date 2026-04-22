from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunConfig:
    name: str
    seed: int
    checkpoint_dir: str
    log_dir: str
    mode: str  # train | play | demo


@dataclass
class GameConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class UMCTSConfig:
    M_s: int
    d_max: int
    c_ucb: float
    dirichlet_alpha: float | None
    dirichlet_frac: float
    rollout_enabled: bool


@dataclass
class NetworkBlockConfig:
    conv_channels: list[int] = field(default_factory=list)
    conv_kernel: int = 3
    mlp_hidden: list[int] = field(default_factory=list)
    activation: str = "relu"


@dataclass
class NNConfig:
    hidden_dim: int
    representation: NetworkBlockConfig
    dynamics: NetworkBlockConfig
    prediction: NetworkBlockConfig
    init_scale: float


@dataclass
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float
    lr_schedule: str


@dataclass
class LossWeights:
    lambda_pi: float
    lambda_v: float
    lambda_r: float


@dataclass
class TrainingConfig:
    N_e: int
    N_es: int
    I_t: int
    gradient_steps_per_training: int
    mbs: int
    q: int
    w: int
    gamma: float
    n_step: int
    optimizer: OptimizerConfig
    loss_weights: LossWeights
    buffer_capacity: int
    # --- optional policy-target knobs ---
    # policy_target_temperature: < 1.0 sharpens the stored policy target so NN_p
    # has a non-uniform signal to fit. 1.0 = raw visit-count distribution.
    policy_target_temperature: float = 1.0
    # q_policy_mix: if > 0, mix a softmax(Q/q_policy_temperature) term into the
    # stored policy target. Lets a well-trained value head pull the policy away
    # from uniform even when raw visit counts are near-uniform.
    q_policy_mix: float = 0.0
    q_policy_temperature: float = 0.5


@dataclass
class LoggingConfig:
    plot_every_train_cycles: int
    checkpoint_every_episodes: int
    log_to_jsonl: bool


@dataclass
class VizConfig:
    pygame_enabled: bool
    pygame_fps: int
    cell_size_px: int
    window_title: str


@dataclass
class Config:
    run: RunConfig
    game: GameConfig
    umcts: UMCTSConfig
    nn: NNConfig
    training: TrainingConfig
    logging: LoggingConfig
    viz: VizConfig
