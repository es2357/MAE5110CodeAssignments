"""
ROA PLOT
: plots regions of attraction from the CSV produced by roa_sim.py
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

DATA_DIR = Path(__file__).resolve().parent / "assignment_1_data"
# Change this filename to choose a different simulation result by default
DEFAULT_CSV = DATA_DIR / "rimless_wheel_roa_gamma_0.390_N_8.csv"

# Each CSV classification maps to (legend text, hex color)
CLASSIFICATIONS = {
    # this is cool - didn't know you could have interactive boxes in python
    "settled": ("Standing (settled)", "#EA4259"),
    "rolling": ("Downhill rolling (limit cycle)", "#30C3EB"),
    "upright_equilibrium": ("Upright equilibrium (unstable)", "#CC1278"),
    "unclassified": ("Unresolved at time limit", "#5011CF"),
    "numerical_failure": ("Numerical failure", "#D55E00"),
}


def load_results(csv_path):
    """Reconstruct the initial-state grid and read its simulation parameters"""
    parameter_names = ("gamma", "gravity", "length", "mass", "num_spokes")
    required_columns = {"theta0", "theta_dot0", "classification", *parameter_names}

    with csv_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing_columns))}")
        rows = list(reader)

    if not rows:
        raise ValueError("The CSV contains no simulation results.")

    # read parameters from the CSV so the title and contact lines match data
    params = {}
    for name in parameter_names:
        values = {float(row[name]) for row in rows}
        if len(values) != 1 or not np.all(np.isfinite(list(values))):
            raise ValueError(f"Expected one finite value of {name} throughout the CSV")
        params[name] = values.pop()

    # np.unique sorts the coordinates, so CSV row order does not matter
    theta_values = np.unique([float(row["theta0"]) for row in rows])
    theta_dot_values = np.unique([float(row["theta_dot0"]) for row in rows])
    for values in (theta_values, theta_dot_values):
        if len(values) < 2 or not np.all(np.isfinite(values)):
            raise ValueError("Each state axis needs at least two finite grid values")

    theta_indices = {value: index for index, value in enumerate(theta_values)}
    theta_dot_indices = {value: index for index, value in enumerate(theta_dot_values)}
    class_codes = {name: index for index, name in enumerate(CLASSIFICATIONS)}

    # rows are angular velocities (y); columns are angles (x)
    # use initial coordinates to reconstruct the basin map
    basin_grid = np.full((len(theta_dot_values), len(theta_values)), -1, dtype=int)

    for row in rows:
        classification = row["classification"]
        if classification not in class_codes:
            raise ValueError(f"Unknown classification: {classification}")

        i = theta_indices[float(row["theta0"])]
        j = theta_dot_indices[float(row["theta_dot0"])]
        if basin_grid[j, i] != -1:
            raise ValueError("The CSV contains duplicate initial conditions.")
        basin_grid[j, i] = class_codes[classification]

    if np.any(basin_grid == -1):
        raise ValueError("The CSV is missing grid points; finish the simulation first.")

    return theta_values, theta_dot_values, basin_grid, params


def find_cell_edges(values):
    """Place cell boundaries halfway between samples, within the sampled range"""
    midpoints = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(([values[0]], midpoints, [values[-1]]))


def plot_regions(theta_values, theta_dot_values, basin_grid, params):
    """Color each initial condition by its long-term behavior."""
    # one solid color per classification
    colors = [color for label, color in CLASSIFICATIONS.values()]
    color_map = ListedColormap(colors)
    normalization = BoundaryNorm(np.arange(len(colors) + 1) - 0.5, len(colors))

    # FIGURE SIZE
    figure, axis = plt.subplots(figsize=(9, 6.5), layout="constrained")
    # Draw colored cells only within the simulated range
    axis.pcolormesh(
        find_cell_edges(theta_values),
        find_cell_edges(theta_dot_values),
        basin_grid,
        cmap=color_map,
        norm=normalization,
        shading="flat",
        rasterized=True,
    )

    # Physical contact angles come from the saved slope and number of spokes
    alpha = np.pi / params["num_spokes"]
    lower_angle = params["gamma"] - alpha
    upper_angle = params["gamma"] + alpha

    # AXIS LABELS
    axis.set_xlabel(r"Initial angle $\theta_0$ (rad)")
    axis.set_ylabel(r"Initial angular velocity $\dot{\theta}_0$ (rad/s)")

    # AXIS RANGES
    axis.set_xlim(theta_values[0], theta_values[-1])
    axis.set_ylim(theta_dot_values[0], theta_dot_values[-1])
    # Avoid solid frame lines filling the gaps in the dashed contact bounds.
    if np.isclose(theta_values[0], lower_angle):
        axis.spines["left"].set_visible(False)
    if np.isclose(theta_values[-1], upper_angle):
        axis.spines["right"].set_visible(False)

    # ZERO REFERENCE LINES
    axis.axhline(0.0, color="0.3", linewidth=0.7, alpha=0.6)
    axis.axvline(0.0, color="0.3", linewidth=0.7, alpha=0.6)

    # CONTACT BOUNDS
    for angle, label, alignment in (
        (lower_angle, rf"$\gamma-\alpha={lower_angle:.3f}$", "left"),
        (upper_angle, rf"$\gamma+\alpha={upper_angle:.3f}$", "right"),
    ):
        axis.axvline(
            angle, color="0.15", linestyle="--", linewidth=1.3,
            zorder=4, clip_on=False,
        )
        axis.annotate(
            label,
            xy=(angle, 0.97),
            xycoords=axis.get_xaxis_transform(),
            xytext=(5 if alignment == "left" else -5, 0),
            textcoords="offset points",
            ha=alignment,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
        )

    # TITLE
    axis.set_title(
        "Rimless wheel: regions of attraction\n"
        rf"$\gamma={params['gamma']:.3f}$ rad, "
        rf"$N={params['num_spokes']:g}$, "
        rf"$g={params['gravity']:g}$ m/s$^2$, "
        rf"$l={params['length']:g}$ m, "
        rf"$m={params['mass']:g}$ kg"
    )

    counts = np.bincount(basin_grid.ravel(), minlength=len(CLASSIFICATIONS))
    handles = [
        Patch(facecolor=color, label=label)
        for code, (label, color) in enumerate(CLASSIFICATIONS.values())
        if counts[code] > 0
    ]
    # LEGEND
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
    )

    return figure


def main():
    # Run python roa_plot.py to generate CSV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=DEFAULT_CSV,
        help="CSV path or filename in assignment_1_data (default: gamma 0.080).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening a plot window.",
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    if csv_path.parent == Path(".") and not csv_path.is_file():
        csv_path = DATA_DIR / csv_path
    try:
        theta_values, theta_dot_values, basin_grid, params = load_results(csv_path)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    figure = plot_regions(theta_values, theta_dot_values, basin_grid, params)
    # SAVE LOCATION
    DATA_DIR.mkdir(exist_ok=True)
    output_path = DATA_DIR / csv_path.with_suffix(".png").name
    figure.savefig(output_path, dpi=300)

    print(f"Grid: {len(theta_values)} angles x {len(theta_dot_values)} velocities")
    counts = np.bincount(basin_grid.ravel(), minlength=len(CLASSIFICATIONS))
    for code, name in enumerate(CLASSIFICATIONS):
        if counts[code] > 0:
            print(f"  {name}: {counts[code]}")
    print(f"Saved plot to: {output_path}")

    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
