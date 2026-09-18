"""
POINCARE RETURN MAP FOR RIMLESS WHEEL
: plots the signed, single-impact Poincare map for 0 <= gamma <= alpha.

Reference: https://underactuated.mit.edu/simple_legs.html
"""

import argparse
import csv
import sys
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import rimless_wheel as model

if __package__ in (None, ""):
    import assignment_1 as simulation
    from roa_plot import CLASSIFICATIONS
else:
    from . import assignment_1 as simulation
    from .roa_plot import CLASSIFICATIONS

PARAMS = model.generate_params()
DATA_DIR = Path(__file__).resolve().parent / "assignment_1_data"

MIN_SPEED = -8.0
MAX_SPEED = 6.0
N_SPEEDS = 701
TIMESTEP = 0.01
MAX_RETURN_TIME = 30.0
IMPACT_TIME_TOL = 1e-9
THRESHOLD_TOL = 1e-12
BASIN_SAMPLES = 28001
MAX_MAP_ITERATIONS = 500
CONVERGENCE_TOL = 1e-8

BASIN_NAMES = {0: "standing", 1: "rolling", -1: "unresolved"}
BASIN_COLORS = {
    0: CLASSIFICATIONS["settled"][1],
    1: CLASSIFICATIONS["rolling"][1],
    -1: CLASSIFICATIONS["unclassified"][1],
}


# c = angular-velocity multiplier at impact
# d = increase in squared angular velocity during one downhill stance
def map_coefficients(params):
    """Return the collision factor and downhill gain in squared angular speed"""
    alpha = np.pi / params["num_spokes"]
    gamma = params["slope_angle"]
    c = np.cos(2 * alpha)
    d = (4 * params["gravity"] / params["length"] * np.sin(alpha) * np.sin(gamma))
    return c, d


def calculate_thresholds(params):
    """Return the critical post-impact velocities (forward, reverse)"""
    lower_angle, upper_angle = model.contact_angles(params)
    scale = 2 * params["gravity"] / params["length"]
    omega_1 = np.sqrt(scale * (1 - np.cos(lower_angle)))
    omega_2 = -np.sqrt(scale * (1 - np.cos(upper_angle)))
    return omega_1, omega_2


def evaluate_return_map(omega, params):
    """Evaluate the three cases for one velocity or an array of velocities"""
    omega = np.asarray(omega, dtype=float)
    c, d = map_coefficients(params)
    omega_1, omega_2 = calculate_thresholds(params)

    # each condition gives True/False for every velocity; & means AND
    # skip nonfinite numbers and speeds within THRESHOLD_TOL of either threshold
    away_from_forward_threshold = np.abs(omega - omega_1) > THRESHOLD_TOL
    away_from_reverse_threshold = np.abs(omega - omega_2) > THRESHOLD_TOL
    valid = np.isfinite(omega) & away_from_forward_threshold & away_from_reverse_threshold

    # select the three intervals in the piecewise equation from the notes
    downhill = valid & (omega > omega_1)
    rocking = valid & (omega_2 < omega) & (omega < omega_1)
    uphill = valid & (omega < omega_2)

    # values we skip stay NaN (undefined)
    values = np.full_like(omega, np.nan)
    values[downhill] = c * np.sqrt(omega[downhill] ** 2 + d)
    values[rocking] = -c * omega[rocking]
    values[uphill] = -c * np.sqrt(omega[uphill] ** 2 - d)
    return values


def calculate_fixed_points(params):
    """Use the reference's fixed-point formulas and existence conditions"""
    alpha = np.pi / params["num_spokes"]
    gamma = params["slope_angle"]
    _, d = map_coefficients(params)
    omega_1, _ = calculate_thresholds(params)
    fixed_points = {}
    if gamma < alpha:
        fixed_points["standing"] = 0.0

    rolling = np.sqrt(d) / np.tan(2 * alpha)
    # rolling solution valid only on the downhill branch
    if rolling > omega_1:
        fixed_points["rolling"] = rolling
    return fixed_points


# Numerical check: simulate one collision
def simulate_return(omega_n, params, timestep=TIMESTEP):
    """Independently simulate the next impact to check the analytic map."""
    result = {
        "omega_n": float(omega_n),
        "omega_next": np.nan,
        "return_time": np.nan,
        "impact_count": 0,
        "return_type": "timeout",
    }

    # At either threshold, including zero when gamma == alpha, no finite impact occurs
    omega_1, omega_2 = calculate_thresholds(params)
    at_forward_threshold = abs(omega_n - omega_1) < THRESHOLD_TOL
    at_reverse_threshold = abs(omega_n - omega_2) < THRESHOLD_TOL
    if at_forward_threshold or at_reverse_threshold:
        result["return_type"] = "upright_asymptote"
        return result

    # Zero is the standing limit, not a simulated flight/impact
    if omega_n == 0:
        result.update(omega_next=0.0, return_type="standing_limit")
        return result

    lower_angle, upper_angle = model.contact_angles(params)
    starting_angle = lower_angle if omega_n > 0 else upper_angle
    state = np.array([starting_angle, omega_n], dtype=float)
    t = 0.0

    while t < MAX_RETURN_TIME:
        dt = min(timestep, MAX_RETURN_TIME - t)
        state, impacts, _ = simulation.step(
            t, state, dt, params,
            impact_time_tol=IMPACT_TIME_TOL,
            stop_at_first_impact=True,
        )

        if not impacts:
            t += dt
            continue

        # Use the first collision's state, before any settling shortcut
        impact = impacts[0]
        direction = impact["direction"]
        if direction.value * omega_n < 0:
            return_type = "rocking_return"
        elif direction == model.ImpactDirection.DOWNHILL:
            return_type = "downhill_step"
        else:
            return_type = "uphill_step"
        result.update(
            omega_next=float(impact["state_plus"][1]), return_time=impact["time"],
            impact_count=1, return_type=return_type,
        )
        return result

    result["return_time"] = t
    return result


def classify_basins(speeds, params, fixed_points):
    """Iterate the map: 0 = standing, 1 = rolling, -1 = unresolved/nonreturn."""
    current = np.asarray(speeds, dtype=float).copy()
    labels = np.full(current.shape, -1, dtype=int)

    for _ in range(MAX_MAP_ITERATIONS):
        # Label velocities that have reached one of the fixed points
        active = (labels == -1) & np.isfinite(current)
        for label, name in ((0, "standing"), (1, "rolling")):
            if name in fixed_points:
                converged = active & (np.abs(current - fixed_points[name]) < CONVERGENCE_TOL)
                labels[converged] = label

        # Advance only the velocities whose outcome is still unknown
        active = (labels == -1) & np.isfinite(current)
        if not np.any(active):
            break
        current[active] = evaluate_return_map(current[active], params)
    return labels


# Plotting

def shade_basins(axis, params, fixed_points):
    """Shade initial velocities by their eventual standing or rolling outcome."""
    basin_speeds = np.linspace(MIN_SPEED, MAX_SPEED, BASIN_SAMPLES)
    basin_labels = classify_basins(basin_speeds, params, fixed_points)
    edges = np.concatenate((
        [MIN_SPEED], 0.5 * (basin_speeds[:-1] + basin_speeds[1:]), [MAX_SPEED],
    ))
    changes = np.flatnonzero(np.diff(basin_labels) != 0) + 1
    boundaries = np.concatenate(([0], changes, [len(basin_labels)]))
    for start, stop in pairwise(boundaries):
        if basin_labels[start] == 1:
            axis.axvspan(
                edges[start], edges[stop], facecolor="none", edgecolor=to_rgba(BASIN_COLORS[1], 0.3),
                hatch="////", linewidth=0, zorder=0,
            )
        elif basin_labels[start] == 0:
            axis.axvspan(
                edges[start], edges[stop], facecolor=to_rgba(BASIN_COLORS[0], 0.08),
                linewidth=0, zorder=0,
            )
        elif basin_labels[start] == -1:
            axis.axvspan(
                edges[start], edges[stop], facecolor=to_rgba(BASIN_COLORS[-1], 0.2),
                linewidth=0, zorder=0,
            )
    return basin_labels


def plot_map_branches(axis, results, params, fixed_points):
    """Draw neighboring samples without connecting across the map's jumps."""
    curve_basins = set()
    for return_type in ("uphill_step", "rocking_return", "downhill_step"):
        included_types = [return_type]
        if return_type == "rocking_return":
            included_types.append("standing_limit")

        segments = []
        midpoints = []
        for left, right in pairwise(results):
            if left["return_type"] not in included_types or right["return_type"] not in included_types:
                continue
            segment = [
                (left["omega_n"], left["omega_next"]),
                (right["omega_n"], right["omega_next"]),
            ]
            if not np.all(np.isfinite(segment)):
                continue
            segments.append(segment)
            midpoints.append(0.5 * (left["omega_n"] + right["omega_n"]))

        if not segments:
            continue

        segment_basins = classify_basins(midpoints, params, fixed_points)
        curve_basins.update(segment_basins.tolist())
        colors = [BASIN_COLORS[label] for label in segment_basins]
        axis.add_collection(LineCollection(segments, colors=colors, linewidths=2.5, zorder=3))
    return curve_basins


def plot_return_map(results, params, fixed_points):
    """Plot the map, its basins, fixed points, and undefined thresholds."""
    omega_1, omega_2 = calculate_thresholds(params)
    settled_color = BASIN_COLORS[0]
    rolling_color = BASIN_COLORS[1]
    unresolved_color = BASIN_COLORS[-1]
    upright_color = CLASSIFICATIONS["upright_equilibrium"][1]

    # Figure size is in inches;
    # MIN_SPEED/MAX_SPEED set both axis limits
    figure, axis = plt.subplots(figsize=(10, 7), layout="constrained")
    basin_labels = shade_basins(axis, params, fixed_points)
    curve_basins = plot_map_branches(axis, results, params, fixed_points)

    axis.plot(
        [MIN_SPEED, MAX_SPEED], [MIN_SPEED, MAX_SPEED],
        color="black", linestyle="--", linewidth=2,
        label="Reference line of slope 1", zorder=2,
    )

    for name, value in fixed_points.items():
        if not MIN_SPEED <= value <= MAX_SPEED:
            continue
        color = settled_color if name == "standing" else rolling_color
        axis.plot(
            value, value, "+", color=color, markersize=14, markeredgewidth=2.5, zorder=5,
        )
        symbol = "stand" if name == "standing" else "roll"
        axis.annotate(
            rf"$\omega^*_{{\mathrm{{{symbol}}}}}={value:.5f}$",
            xy=(value, value), xytext=(-55, -40) if name == "standing" else (15, -42),
            textcoords="offset points", color="0.15", fontsize=12,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2},
            arrowprops={"arrowstyle": "-", "color": color},
        )

    for index, (threshold, direction) in enumerate(((omega_2, "rev"), (omega_1, "fwd"))):
        if MIN_SPEED <= threshold <= MAX_SPEED:
            axis.axvline(
                threshold, color=upright_color, linestyle="--", linewidth=1.8,
                label="Critical post-impact angular velocities" if index == 0 else None,
                zorder=3,
            )
            axis.text(
                threshold + (-0.15 if index == 0 else 0.15),
                MAX_SPEED - 0.055 * (MAX_SPEED - MIN_SPEED),
                rf"$\omega_{{\mathrm{{crit,{direction}}}}}$", color=upright_color, fontsize=15,
                ha="right" if index == 0 else "left",
            )

    c, d = map_coefficients(params)
    reverse_limits = (
        -c * np.sqrt(max(0, omega_2**2 - d)),
        -c * omega_2,
    )
    forward_limits = (
        -c * omega_1,
        c * np.sqrt(omega_1**2 + d),
    )
    for threshold, values in (
        (omega_2, reverse_limits),
        (omega_1, forward_limits),
    ):
        if MIN_SPEED <= threshold <= MAX_SPEED:
            axis.plot(
                [threshold, threshold], values, linestyle="none", marker="o",
                markerfacecolor="white", markeredgecolor=upright_color, markersize=5, zorder=4,
            )

    axis.set_xlabel(r"Angular velocity after collision $n$, $\dot{\theta}_n^+$ (rad/s)")
    axis.set_ylabel(r"Angular velocity after collision $n+1$, $\dot{\theta}_{n+1}^+$ (rad/s)")
    axis.set_xlim(MIN_SPEED, MAX_SPEED)
    axis.set_ylim(MIN_SPEED, MAX_SPEED)
    axis.set_title(
        "Rimless wheel: Poincaré return map\n"
        rf"$\gamma={params['slope_angle']:.3f}$ rad, $N={params['num_spokes']}$, "
        rf"$g={params['gravity']:g}$ m/s$^2$, $l={params['length']:g}$ m"
    )
    axis.grid(color="0.7", linestyle=":", linewidth=0.7, zorder=1)
    handles, labels = axis.get_legend_handles_labels()
    for code, name in BASIN_NAMES.items():
        if code in curve_basins:
            handles.append(Line2D([0], [0], color=BASIN_COLORS[code], linewidth=2.5))
            labels.append(f"Return map: {name} basin")
    if fixed_points:
        handles.append(Line2D(
            [0], [0], color="black", marker="+", markersize=12,
            markeredgewidth=2, linestyle="none",
        ))
        labels.append("Fixed points (colored by attractor)")
    if np.any(basin_labels == 1):
        handles.append(Patch(facecolor="white", edgecolor=to_rgba(rolling_color, 0.3), hatch="////"))
        labels.append("Hatched: converges to rolling")
    if np.any(basin_labels == 0):
        handles.append(Patch(facecolor=to_rgba(settled_color, 0.08), edgecolor=settled_color))
        labels.append("Pink shading: converges to standing")
    if np.any(basin_labels == -1):
        handles.append(Patch(facecolor=to_rgba(unresolved_color, 0.2), edgecolor=unresolved_color))
        labels.append("Purple shading: unresolved / no return")
    axis.legend(handles, labels, loc="lower right", fontsize=9, framealpha=1)
    return figure


# Sampling and output

def sample_velocities(params, fixed_points):
    """Sample the speed range, including fixed points and both sides of each jump."""
    speeds = list(np.linspace(MIN_SPEED, MAX_SPEED, N_SPEEDS))
    extra_speeds = [0.0]
    extra_speeds.extend(fixed_points.values())
    for threshold in calculate_thresholds(params):
        extra_speeds.extend([threshold - 1e-4, threshold, threshold + 1e-4])

    for speed in extra_speeds:
        if MIN_SPEED <= speed <= MAX_SPEED:
            speeds.append(speed)
    return np.unique(speeds)  # Sort the velocities and remove duplicates.


def save_results(results, params, csv_path):
    """Save the simulated returns and the parameters used for each row."""
    omega_1, omega_2 = calculate_thresholds(params)
    parameter_columns = {
        "gamma": params["slope_angle"],
        "alpha": np.pi / params["num_spokes"],
        "omega_1": omega_1,
        "omega_2": omega_2,
        "gravity": params["gravity"],
        "length": params["length"],
        "mass": params["mass"],
        "num_spokes": params["num_spokes"],
    }
    columns = list(results[0]) + list(parameter_columns)
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for result in results:
            row = result.copy()
            row.update(parameter_columns)
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Save without opening a plot window.")
    args = parser.parse_args()

    params = PARAMS.copy()
    gamma = params["slope_angle"]
    omega_1, omega_2 = calculate_thresholds(params)
    fixed_points = calculate_fixed_points(params)
    speeds = sample_velocities(params, fixed_points)

    results = [simulate_return(speed, params) for speed in speeds]
    exact_values = evaluate_return_map(speeds, params)
    basins = classify_basins(speeds, params, fixed_points)
    for row, exact, basin in zip(results, exact_values, basins, strict=True):
        row["omega_next_exact"] = float(exact)
        row["absolute_error"] = abs(row["omega_next"] - exact)
        row["attractor"] = BASIN_NAMES[basin]

    DATA_DIR.mkdir(exist_ok=True)
    output_stem = DATA_DIR / f"rimless_wheel_return_map_gamma_{gamma:.3f}"
    csv_path = Path(f"{output_stem}.csv")
    save_results(results, params, csv_path)

    figure = plot_return_map(results, params, fixed_points)
    image_path = Path(f"{output_stem}.png")
    figure.savefig(image_path, dpi=300)

    print("Map convention: signed post-impact velocity at consecutive collisions")
    print(f"omega_critical_fwd = {omega_1:.8f}, omega_critical_reverse = {omega_2:.8f} rad/s (undefined inputs)")
    for name, value in fixed_points.items():
        print(f"{name.capitalize()} fixed point: {value:.8f} rad/s")
    if "rolling" not in fixed_points:
        print("No rolling fixed point exists for these parameters.")
    finite_errors = [row["absolute_error"] for row in results if np.isfinite(row["absolute_error"])]
    if finite_errors:
        print(f"Maximum simulation/formula difference: {max(finite_errors):.3g} rad/s")
    for return_type in sorted({row["return_type"] for row in results}):
        count = sum(row["return_type"] == return_type for row in results)
        print(f"  {return_type}: {count}")
    print(f"Saved data to: {csv_path}")
    print(f"Saved plot to: {image_path}")

    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
