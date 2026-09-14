"""
ROA SIMULATION
: chooses starting states, classifies their behavior, and saves results;
estimates the rimless wheel's region of attraction over initial states

rimless_wheel.py supplies the parameters and contact angles
assignment_1.py supplies the timestep, impact handling, and settling test
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

# should maybe reconsider how I structured the code...
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import assignment_1 as simulation
else:
    from . import assignment_1 as simulation

from models import rimless_wheel as model

DATA_DIR = Path(__file__).resolve().parent / "assignment_1_data"

# number of starting angles and angular velocities to test
NUM_INITIAL_ANGLES = 81
NUM_INITIAL_ANGULAR_VELOCITIES = 161

# simulation timing (seconds)
SIMULATION_TIME_STEP = 0.01
MAX_SIMULATION_TIME = 30.0

# Check whether recent downhill impacts have repeatable speeds
NUM_DOWNHILL_IMPACTS = 6
POST_IMPACT_SPEED_TOLERANCE = 1e-4  # (rad/s)

# Precision when locating an impact within a timestep
IMPACT_TIME_TOLERANCE = 1e-9  # (seconds)

# Skip any leftover timestep this small or smaller
MIN_REMAINING_TIME = 1e-12  # (seconds)


def rolling_has_converged(post_impact_speeds):
    """
    Determine whether the downhill post-impact angular
    velocities have converged to a repeating value
    """

    if len(post_impact_speeds) < NUM_DOWNHILL_IMPACTS:
        return False

    recent = np.array(post_impact_speeds[-NUM_DOWNHILL_IMPACTS:])
    if np.any(recent <= 0.0):
        return False
    differences = np.abs(np.diff(recent))

    return np.max(differences) < POST_IMPACT_SPEED_TOLERANCE


def simulate_initial_condition(theta0, theta_dot0, params):
    """Classify one initial state using the rimless-wheel simulation"""

    state = np.array([theta0, theta_dot0], dtype=float)
    t = 0.0

    downhill_post_impact_speeds = []

    total_impacts = 0
    downhill_impacts = 0
    uphill_impacts = 0

    # exact unstable upright equilibrium
    if (abs(theta0) < 1e-12 and abs(theta_dot0) < 1e-12):
        return {
            "classification": "upright_equilibrium",
            "final_time": 0.0,
            "final_theta": theta0,
            "final_theta_dot": theta_dot0,
            "total_impacts": 0,
            "downhill_impacts": 0,
            "uphill_impacts": 0,
            "rolling_speed": np.nan,
        }

    # simulate
    while t < MAX_SIMULATION_TIME:

        dt = min(SIMULATION_TIME_STEP, MAX_SIMULATION_TIME - t)

        state, impacts, settled = simulation.step(
            t,
            state,
            dt,
            params,
            impact_time_tol=IMPACT_TIME_TOLERANCE,
            remaining_time_tol=MIN_REMAINING_TIME,
        )

        t += dt

        # process impacts that occurred during this timestep
        for impact in impacts:
            total_impacts += 1

            if impact["direction"] == model.ImpactDirection.DOWNHILL:

                downhill_impacts += 1
                omega_plus = impact["state_plus"][1]
                downhill_post_impact_speeds.append(omega_plus)

            elif impact["direction"] == model.ImpactDirection.UPHILL:
                uphill_impacts += 1

        # Outcome 1: wheel settles / cannot continue rolling
        if settled:
            return {
                "classification": "settled",
                "final_time": t,
                "final_theta": state[0],
                "final_theta_dot": state[1],
                "total_impacts": total_impacts,
                "downhill_impacts": downhill_impacts,
                "uphill_impacts": uphill_impacts,
                "rolling_speed": np.nan,
            }

        # Outcome 2: stable downhill rolling limit cycle
        if rolling_has_converged(
            downhill_post_impact_speeds
        ):
            rolling_speed = np.mean(downhill_post_impact_speeds[-NUM_DOWNHILL_IMPACTS:])
            return {
                "classification": "rolling",
                "final_time": t,
                "final_theta": state[0],
                "final_theta_dot": state[1],
                "total_impacts": total_impacts,
                "downhill_impacts": downhill_impacts,
                "uphill_impacts": uphill_impacts,
                "rolling_speed": rolling_speed,
            }

        # numerical safety check
        if not np.all(np.isfinite(state)):
            return {
                "classification": "numerical_failure",
                "final_time": t,
                "final_theta": state[0],
                "final_theta_dot": state[1],
                "total_impacts": total_impacts,
                "downhill_impacts": downhill_impacts,
                "uphill_impacts": uphill_impacts,
                "rolling_speed": np.nan,
            }

    # simulation reached MAX_SIMULATION_TIME without classification
    return {
        "classification": "unclassified",
        "final_time": t,
        "final_theta": state[0],
        "final_theta_dot": state[1],
        "total_impacts": total_impacts,
        "downhill_impacts": downhill_impacts,
        "uphill_impacts": uphill_impacts,
        "rolling_speed": np.nan,
    }


def main():
    """Run the initial-condition grid and save its classifications to CSV"""
    params = model.generate_params()
    gamma = params["slope_angle"]
    alpha = np.pi / params["num_spokes"]
    lower_angle, upper_angle = model.contact_angles(params)

    print(f"gamma = {gamma:.4f} rad")
    print(f"alpha = {alpha:.4f} rad")
    print(f"theta range = [{lower_angle:.4f}, {upper_angle:.4f}] rad")

    theta_values = np.linspace(lower_angle, upper_angle, NUM_INITIAL_ANGLES)
    theta_dot_values = np.linspace(-3.0, 3.0, NUM_INITIAL_ANGULAR_VELOCITIES)

    filename = DATA_DIR / f"rimless_wheel_roa_gamma_{gamma:.3f}.csv"
    fieldnames = [
        "theta_index",
        "theta_dot_index",
        "theta0",
        "theta_dot0",
        "classification",
        "final_time",
        "final_theta",
        "final_theta_dot",
        "total_impacts",
        "downhill_impacts",
        "uphill_impacts",
        "rolling_speed",
        "gamma",
        "alpha",
        "gravity",
        "length",
        "mass",
        "num_spokes",
    ]

    total_simulations = len(theta_values) * len(theta_dot_values)
    simulation_number = 0
    classification_counts = {}
    start_time = time.perf_counter()

    DATA_DIR.mkdir(exist_ok=True)
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i_theta, theta0 in enumerate(theta_values):
            for i_theta_dot, theta_dot0 in enumerate(theta_dot_values):
                result = simulate_initial_condition(theta0, theta_dot0, params)

                classification = result["classification"]
                classification_counts[classification] = (
                    classification_counts.get(classification, 0) + 1
                )

                writer.writerow({
                    "theta_index": i_theta,
                    "theta_dot_index": i_theta_dot,
                    "theta0": theta0,
                    "theta_dot0": theta_dot0,
                    **result,
                    "gamma": gamma,
                    "alpha": alpha,
                    "gravity": params["gravity"],
                    "length": params["length"],
                    "mass": params["mass"],
                    "num_spokes": params["num_spokes"],
                })

                simulation_number += 1
                if (
                    simulation_number % 250 == 0
                    or simulation_number == total_simulations
                ):
                    elapsed = time.perf_counter() - start_time
                    percent = 100.0 * simulation_number / total_simulations
                    print(
                        f"{simulation_number:5d} / {total_simulations:5d} "
                        f"({percent:5.1f}%)  elapsed = {elapsed:.1f} s"
                    )

    elapsed = time.perf_counter() - start_time
    print()
    print("---------------------------------------")
    print("RoA simulations complete")
    print("---------------------------------------")
    print(f"Saved data to: {filename}")
    print(f"Total runtime: {elapsed:.2f} s")
    print()
    print("Classification counts:")
    for classification, count in classification_counts.items():
        print(f"  {classification:20s}: {count}")


if __name__ == "__main__":
    main()
