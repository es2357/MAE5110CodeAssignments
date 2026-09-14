"""
FLOQUET MULTIPLIER ESTIMATION
: estimates how much a small speed error shrinks after one rimless-wheel step
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import rimless_wheel_return_map as return_map
else:
    from . import rimless_wheel_return_map as return_map

from models import rimless_wheel as model

DATA_DIR = return_map.DATA_DIR
MULTIPLIER_TOL = 1e-4


def simulate_downhill_return(omega, params, timestep):
    """Return the speed just after the next impact"""
    result = return_map.simulate_return(omega, params, timestep=timestep)
    return result["omega_next"]


def estimate_floquet(params, timestep=return_map.TIMESTEP, epsilon=None):
    """Estimate P'(omega_star) with smaller and smaller speed perturbations"""
    alpha = np.pi / params["num_spokes"]

    # find the steady speed and keep both perturbations on the rolling branch
    omega_star = return_map.calculate_fixed_points(params)["rolling"]
    omega_critical, _ = return_map.calculate_thresholds(params)
    rolling_margin = omega_star - omega_critical

    # start at 1% of steady speed, or smaller near the rolling threshold
    if epsilon is None:
        epsilon = min(0.01 * omega_star, 0.25 * rolling_margin)

    next_at_fixed_point = simulate_downhill_return(omega_star, params, timestep)
    exact_multiplier = float(np.cos(2 * alpha) ** 2)

    # simulate above and below steady speed, then measure the return-map slopes
    results = []
    for index in range(5):
        perturbation = epsilon / 2**index
        omega_below = omega_star - perturbation
        omega_above = omega_star + perturbation

        next_below = simulate_downhill_return(omega_below, params, timestep)
        next_above = simulate_downhill_return(omega_above, params, timestep)

        # Rise / run between the two simulated returns.
        multiplier = (next_above - next_below) / (2 * perturbation)
        results.append({
            "epsilon": perturbation,
            "omega_below": omega_below,
            "next_below": next_below,
            "omega_above": omega_above,
            "next_above": next_above,
            "slope_below": (next_at_fixed_point - next_below) / perturbation,
            "slope_above": (next_above - next_at_fixed_point) / perturbation,
            "multiplier": multiplier,
            "absolute_error": abs(multiplier - exact_multiplier),
        })

    # Trust the estimate only if it converges and simulation error is small
    last, previous = results[-1], results[-2]
    residual = next_at_fixed_point - omega_star
    resolved = (
        last["absolute_error"] < MULTIPLIER_TOL
        and abs(last["multiplier"] - previous["multiplier"]) < MULTIPLIER_TOL
        and abs(residual) < 0.01 * last["epsilon"]
    )
    return results, {
        "omega_star": omega_star,
        "next_at_fixed_point": next_at_fixed_point,
        "fixed_point_residual": residual,
        "exact_multiplier": exact_multiplier,
        "timestep": timestep,
        **params,
        "estimate_resolved": bool(resolved),
    }


def plot_estimates(results, metadata):
    """Show the sampled local return map and perturbation-size convergence."""
    figure, (map_axis, convergence_axis) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    omega_star = metadata["omega_star"]
    multiplier = results[-1]["multiplier"]
    span = 1.15 * results[0]["epsilon"]
    speeds = np.array([omega_star - span, omega_star + span])
    map_axis.plot(speeds, speeds, "k--", label="Next = current")
    map_axis.plot(
        speeds,
        metadata["next_at_fixed_point"] + multiplier * (speeds - omega_star),
        label=f"Estimated slope = {multiplier:.6f}",
    )
    map_axis.plot(
        [row["omega_below"] for row in results], [row["next_below"] for row in results],
        "o", label="Below steady speed",
    )
    map_axis.plot(
        [row["omega_above"] for row in results], [row["next_above"] for row in results],
        "o", label="Above steady speed",
    )
    map_axis.plot(omega_star, omega_star, "k*", markersize=10, label="Steady rolling")
    map_axis.set(
        xlabel=r"Current post-impact velocity $\omega_n$ (rad/s)",
        ylabel=r"Next post-impact velocity $P(\omega_n)$ (rad/s)",
        title="Return map near steady rolling",
    )
    convergence_axis.semilogx(
        [row["epsilon"] for row in results],
        [row["multiplier"] for row in results],
        "o-", label="Central difference from simulations",
    )
    convergence_axis.axhline(
        metadata["exact_multiplier"], color="black", linestyle="--",
        label=f"Analytic check = {metadata['exact_multiplier']:.6f}",
    )
    convergence_axis.invert_xaxis()
    convergence_axis.set(
        xlabel=r"Perturbation $\varepsilon$ (rad/s), decreasing to the right",
        ylabel=r"Estimated Floquet multiplier $\lambda$",
        title="Convergence as perturbations shrink",
    )
    for axis in (map_axis, convergence_axis):
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        axis.ticklabel_format(axis="y", useOffset=False)
    map_axis.ticklabel_format(axis="x", useOffset=False)
    figure.suptitle(
        f"Rimless wheel: N = {metadata['num_spokes']}, "
        f"gamma = {metadata['slope_angle']:.4f} rad"
    )
    return figure


def main():
    params = model.generate_params()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=params["slope_angle"],
                        help="Slope inclination in radians.")
    parser.add_argument("--spokes", type=int, default=params["num_spokes"],
                        help="Number of spokes (greater than four).")
    parser.add_argument("--timestep", type=float, default=return_map.TIMESTEP,
                        help="Integration timestep in seconds.")
    parser.add_argument("--epsilon", type=float,
                        help="Initial perturbation in rad/s; halved four times.")
    parser.add_argument("--no-show", action="store_true",
                        help="Save without opening a plot window.")
    args = parser.parse_args()
    params.update(slope_angle=args.gamma, num_spokes=args.spokes)
    results, metadata = estimate_floquet(params, args.timestep, args.epsilon)

    print("Map: immediately after one impact to immediately after the next.")
    print(f"Rolling fixed point: {metadata['omega_star']:.10f} rad/s")
    print(f"Fixed-point simulation residual: {metadata['fixed_point_residual']:.3e} rad/s")
    print(f"Analytic multiplier: {metadata['exact_multiplier']:.10f}")
    print("\n   epsilon       slope below     slope above     central slope    abs. error")
    for row in results:
        print(
            f"{row['epsilon']:12.5e}  {row['slope_below']:14.9f}  "
            f"{row['slope_above']:14.9f}  {row['multiplier']:14.9f}  "
            f"{row['absolute_error']:10.3e}"
        )
    multiplier = results[-1]["multiplier"]
    print(f"\nFloquet estimate at smallest epsilon: {multiplier:.8f}")
    if not metadata["estimate_resolved"]:
        print(
            "Estimate is not numerically resolved; no stability conclusion. "
            "Try a larger epsilon or smaller timestep."
        )
    elif abs(multiplier) < 1:
        print(f"Locally stable: small speed errors shrink by a factor of {abs(multiplier):.6f} per step.")
    elif abs(multiplier) > 1:
        print("Locally unstable: small speed errors grow each step.")
    else:
        print("Unit multiplier: the linear estimate is inconclusive.")

    DATA_DIR.mkdir(exist_ok=True)
    stem = DATA_DIR / f"floquet_gamma_{args.gamma:.6f}_N_{args.spokes}"
    csv_path = Path(f"{stem}.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        rows = [{**row, **metadata} for row in results]
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    figure = plot_estimates(results, metadata)
    image_path = Path(f"{stem}.png")
    figure.savefig(image_path, dpi=200)
    print(f"Saved data to: {csv_path}")
    print(f"Saved plot to: {image_path}")
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
