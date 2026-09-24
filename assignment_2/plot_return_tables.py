"""
Return tables

"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from assignment_2 import build_control_table, params, project_root


def load_table(output):
    """Return the 641-row return table, building and saving it if needed."""
    settings = {"grid": [33, 49, 641, 5], "timestep": 0.001, "params": params}
    path = output / "return_table_641.npz"
    if path.exists():
        with np.load(path) as saved:
            if json.loads(str(saved["settings"])) == settings:
                print(f"Using saved table: {path}", flush=True)
                return {key: saved[key] for key in saved.files if key != "settings"}

    _, table = build_control_table(settings["grid"], settings["timestep"])
    table["outcomes"] = table["outcomes"].astype(str)
    np.savez(path, **table, settings=json.dumps(settings))
    return table


def plot_tables(table, output):
    """Save the coarse-to-fine return-table picture and return its path."""
    styles = [
        ("captured", "Reached standing RoA", "green"),
        ("returned", "Returned to midstance", "royalblue"),
        ("failed", "Failed trial", "tomato"),
        ("unresolved", "Time limit reached", "lightgray"),
    ]
    colors = ListedColormap([color for _, _, color in styles])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharex=True, sharey=True)

    # These grids share samples, so each simulated table entry can be reused.
    for ax, count in zip(axes, [11, 321, 641]):
        stride = 640 // (count - 1)
        speeds = table["speed_values"][::stride]
        outcomes = table["outcomes"][::stride]
        codes = np.full(outcomes.shape, np.nan)
        for code, (outcome, _, _) in enumerate(styles):
            codes[outcomes == outcome] = code
        ax.pcolormesh(table["alpha_values"], speeds, codes, cmap=colors,
                      vmin=-0.5, vmax=3.5, shading="nearest")
        ax.set_title(f"{count} velocities × 5 angles")
        ax.set_xlabel(r"Foot-placement angle $\alpha$ (rad)")
        ax.set_xlim(table["alpha_values"][[0, -1]])
        ax.set_ylim(speeds[[0, -1]])

    axes[0].set_ylabel(r"Initial midstance velocity $\dot{\theta}$ (rad/s)")
    fig.suptitle("Effect of velocity resolution on the return table")
    fig.legend(handles=[Patch(color=color, label=label) for _, label, color in styles],
               loc="lower center", ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    path = output / "return_table_resolution.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


if __name__ == "__main__":
    plt.switch_backend("Agg")
    output = project_root / "output" / "assignment_2" / "grid_comparison"
    output.mkdir(parents=True, exist_ok=True)
    print(f"Saved comparison to {plot_tables(load_table(output), output)}")
