from pathlib import Path

import yaml

from configs._schema import (
    Config,
    GameConfig,
    LoggingConfig,
    LossWeights,
    NetworkBlockConfig,
    NNConfig,
    OptimizerConfig,
    RunConfig,
    TrainingConfig,
    UMCTSConfig,
    VizConfig,
)


def _block(d: dict) -> NetworkBlockConfig:
    return NetworkBlockConfig(
        conv_channels=list(d.get("conv_channels", [])),
        conv_kernel=int(d.get("conv_kernel", 3)),
        mlp_hidden=list(d.get("mlp_hidden", [])),
        activation=str(d.get("activation", "relu")),
    )


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    run = RunConfig(**raw["run"])
    game = GameConfig(name=raw["game"]["name"], params=dict(raw["game"].get("params") or {}))
    umcts = UMCTSConfig(**raw["umcts"])

    nn_raw = raw["nn"]
    nn = NNConfig(
        hidden_dim=int(nn_raw["hidden_dim"]),
        representation=_block(nn_raw["representation"]),
        dynamics=_block(nn_raw["dynamics"]),
        prediction=_block(nn_raw["prediction"]),
        init_scale=float(nn_raw.get("init_scale", 1.0)),
    )

    tr = raw["training"]
    training = TrainingConfig(
        N_e=int(tr["N_e"]),
        N_es=int(tr["N_es"]),
        I_t=int(tr["I_t"]),
        gradient_steps_per_training=int(tr["gradient_steps_per_training"]),
        mbs=int(tr["mbs"]),
        q=int(tr["q"]),
        w=int(tr["w"]),
        gamma=float(tr["gamma"]),
        n_step=int(tr["n_step"]),
        optimizer=OptimizerConfig(**tr["optimizer"]),
        loss_weights=LossWeights(**tr["loss_weights"]),
        buffer_capacity=int(tr["buffer_capacity"]),
        policy_target_temperature=float(tr.get("policy_target_temperature", 1.0)),
        q_policy_mix=float(tr.get("q_policy_mix", 0.0)),
        q_policy_temperature=float(tr.get("q_policy_temperature", 0.5)),
    )

    logging = LoggingConfig(**raw["logging"])
    viz = VizConfig(**raw["viz"])

    return Config(run=run, game=game, umcts=umcts, nn=nn, training=training,
                  logging=logging, viz=viz)
