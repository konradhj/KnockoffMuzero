from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrainingDashboard:
    """Maintains running time-series and dumps a 4-panel png on each flush."""

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.loss = []
        self.loss_pi = []
        self.loss_v = []
        self.loss_r = []
        self.episode_return = []
        self.policy_entropy = []
        self.value_error = []

    def add_train_step(self, metrics: dict) -> None:
        self.loss.append(metrics.get("loss", float("nan")))
        self.loss_pi.append(metrics.get("loss_pi", float("nan")))
        self.loss_v.append(metrics.get("loss_v", float("nan")))
        self.loss_r.append(metrics.get("loss_r", float("nan")))

    def add_episode(self, total_return: float, mean_entropy: float,
                    value_err: float) -> None:
        self.episode_return.append(total_return)
        self.policy_entropy.append(mean_entropy)
        self.value_error.append(value_err)

    def flush(self) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax = axes[0, 0]
        ax.plot(self.loss, label="total")
        ax.plot(self.loss_pi, label="policy")
        ax.plot(self.loss_v, label="value")
        ax.plot(self.loss_r, label="reward")
        ax.set_title("Loss (per gradient step)")
        ax.set_xlabel("gradient step")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(self.episode_return, color="tab:green")
        ax.set_title("Episode return")
        ax.set_xlabel("episode")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(self.policy_entropy, color="tab:purple")
        ax.set_title("Mean root-policy entropy")
        ax.set_xlabel("episode")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(self.value_error, color="tab:red")
        ax.set_title("|v* - actual return| at root")
        ax.set_xlabel("episode")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.out_dir / "training_dashboard.png", dpi=110)
        plt.close(fig)
