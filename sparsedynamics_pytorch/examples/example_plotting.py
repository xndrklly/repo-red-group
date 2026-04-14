"""
Small plotting helpers for example scripts.
"""

from pathlib import Path

import matplotlib.pyplot as plt


def figures_dir() -> Path:
    """Return the repository-local output directory for example figures."""
    return Path(__file__).resolve().parents[1] / "figures"


def save_loss_plot(
    histories: dict[str, list[float]],
    output_path: Path,
    title: str,
    ylabel: str = "MSE loss",
) -> Path:
    """Save a log-scale loss-vs-epoch plot for one or more training histories."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for label, values in histories.items():
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, label=label, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    if len(histories) > 1:
        ax.legend()

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
