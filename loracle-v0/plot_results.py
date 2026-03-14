"""Plot training loss curve and evaluation results for LoRAcle v0.

Produces the main figure for the experiment report.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

def main():
    base_dir = Path(__file__).parent
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Load loss log
    loss_path = base_dir / "interpreter_loss_log.json"
    with open(loss_path) as f:
        loss_log = json.load(f)

    epochs = [entry["epoch"] for entry in loss_log]
    losses = [entry["loss"] for entry in loss_log]

    ax.plot(epochs, losses, color="#2563eb", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Interpreter LoRA Training Loss\n(2 topics: Harry Potter vs Quantum Physics)", fontsize=13)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Annotate start and end
    ax.annotate(
        f"Epoch 1: {losses[0]:.2f}",
        xy=(epochs[0], losses[0]),
        xytext=(30, 10),
        textcoords="offset points",
        fontsize=10,
        color="#666",
        arrowprops=dict(arrowstyle="->", color="#999"),
    )
    ax.annotate(
        f"Epoch {epochs[-1]}: {losses[-1]:.3f}",
        xy=(epochs[-1], losses[-1]),
        xytext=(-80, 30),
        textcoords="offset points",
        fontsize=10,
        color="#666",
        arrowprops=dict(arrowstyle="->", color="#999"),
    )

    plt.tight_layout()
    out_path = base_dir / "training_loss.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
