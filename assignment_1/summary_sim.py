"""Collect comparable RoA grids and numerical Floquet estimates for the summaries.

Each case samples the same relative contact angle and initial-speed range.
Existing 81 x 161 grids are checked before selecting every fourth state;
missing cases use the same simulator on a 21 x 41 grid.
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import floquet
    import rimless_wheel_return_map as return_map
    import roa_sim
else:
    from . import floquet, roa_sim
    from . import rimless_wheel_return_map as return_map

model = roa_sim.model
DATA_DIR = Path(__file__).resolve().parent / "assignment_1_data"
SUMMARY_CSV = DATA_DIR / "rimless_wheel_parameter_summary.csv"
GAMMAS = (0.04, 0.08, 0.12, 0.16, 0.20, 0.26, 0.39)
SPOKE_COUNTS = range(6, 13)
NUM_ANGLES, NUM_SPEEDS = 21, 41
SETTINGS = {
    "num_angles": NUM_ANGLES,
    "num_speeds": NUM_SPEEDS,
    "minimum_speed": -3.0,
    "maximum_speed": 3.0,
    "timestep": 0.01,
    "max_time": 30.0,
    "retry_time": 60.0,
    "impact_time_tolerance": 1e-9,
    "remaining_time_tolerance": 1e-12,
    "convergence_impacts": 6,
    "speed_tolerance": 1e-4,
    "cache_version": 1,
}
CLASSIFICATIONS = {
    "rolling", "settled", "upright_equilibrium", "unclassified", "numerical_failure",
}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parameter_metadata(params):
    return {
        "gamma": params["slope_angle"],
        "alpha": np.pi / params["num_spokes"],
        "gravity": params["gravity"],
        "length": params["length"],
        "mass": params["mass"],
        "num_spokes": params["num_spokes"],
    }


def read_grid(path, params, num_angles, num_speeds, check_settings=False):
    """Accept only a complete grid whose coordinates and metadata match."""
    if not path.exists():
        return None
    metadata = parameter_metadata(params)
    if check_settings:
        metadata.update(SETTINGS)
    angles = np.linspace(*model.contact_angles(params), num_angles)
    speeds = np.linspace(SETTINGS["minimum_speed"], SETTINGS["maximum_speed"], num_speeds)
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            first = next(reader)
            # Most existing files belong to a different parameter case.
            if any(not np.isclose(float(first[key]), value, rtol=1e-12, atol=0)
                   for key, value in metadata.items()):
                return None
            rows = [first, *reader]
        if len(rows) != num_angles * num_speeds:
            return None
        seen = set()
        for row in rows:
            i, j = int(row["theta_index"]), int(row["theta_dot_index"])
            if not (0 <= i < num_angles and 0 <= j < num_speeds) or (i, j) in seen:
                return None
            seen.add((i, j))
            if any(not np.isclose(float(row[key]), value, rtol=1e-12, atol=0)
                   for key, value in metadata.items()):
                return None
            if not (np.isclose(float(row["theta0"]), angles[i], rtol=0, atol=1e-12)
                    and np.isclose(float(row["theta_dot0"]), speeds[j], rtol=0, atol=1e-12)):
                return None
            if row["classification"] not in CLASSIFICATIONS:
                return None
            if not 0 <= float(row["final_time"]) <= SETTINGS["retry_time"] + 1e-8:
                return None
            if any(int(row[key]) < 0 for key in ("total_impacts", "downhill_impacts", "uphill_impacts")):
                return None
            if int(row["total_impacts"]) != int(row["downhill_impacts"]) + int(row["uphill_impacts"]):
                return None
            if row["classification"] != "numerical_failure" and not all(
                np.isfinite(float(row[key])) for key in ("final_theta", "final_theta_dot")
            ):
                return None
            if row["classification"] == "rolling" and not np.isfinite(float(row["rolling_speed"])):
                return None
            if check_settings:
                if not np.isclose(float(row["relative_angle"]),
                                  (angles[i] - params["slope_angle"]) / metadata["alpha"],
                                  rtol=0, atol=1e-12):
                    return None
                float(row["original_grid_rolling_percent"])
                if not row["original_source_csv"]:
                    return None
        return sorted(rows, key=lambda row: (int(row["theta_index"]), int(row["theta_dot_index"])))
    except (KeyError, ValueError, StopIteration, csv.Error):
        return None


def configure_simulator():
    """Use explicit, recorded settings instead of editable script defaults."""
    roa_sim.SIMULATION_TIME_STEP = SETTINGS["timestep"]
    roa_sim.MAX_SIMULATION_TIME = SETTINGS["max_time"]
    roa_sim.IMPACT_TIME_TOLERANCE = SETTINGS["impact_time_tolerance"]
    roa_sim.MIN_REMAINING_TIME = SETTINGS["remaining_time_tolerance"]
    roa_sim.NUM_DOWNHILL_IMPACTS = SETTINGS["convergence_impacts"]
    roa_sim.POST_IMPACT_SPEED_TOLERANCE = SETTINGS["speed_tolerance"]


def simulate_state(theta, speed, params, time_limit):
    roa_sim.MAX_SIMULATION_TIME = time_limit
    try:
        return roa_sim.simulate_initial_condition(theta, speed, params)
    except RuntimeError:
        # An impact-loop failure must not be silently counted as an attractor.
        return {
            "classification": "numerical_failure", "final_time": 0.0,
            "final_theta": theta, "final_theta_dot": speed,
            "total_impacts": 0, "downhill_impacts": 0, "uphill_impacts": 0,
            "rolling_speed": np.nan,
        }


def collect_grid(params):
    gamma, spokes = params["slope_angle"], params["num_spokes"]
    path = DATA_DIR / f"summary_roa_gamma_{gamma:.3f}_N_{spokes}_grid_21x41.csv"
    cached = read_grid(path, params, NUM_ANGLES, NUM_SPEEDS, check_settings=True)
    if cached is not None:
        return cached, path

    source, original_percent, original_rows = "new simulation", np.nan, None
    for candidate in sorted(DATA_DIR.glob("rimless_wheel_roa_gamma_*.csv")):
        original_rows = read_grid(candidate, params, 81, 161)
        if original_rows is not None:
            source = candidate.name
            original_percent = 100 * sum(row["classification"] == "rolling" for row in original_rows) / len(original_rows)
            break
    original = {}
    if original_rows is not None:
        original = {(int(row["theta_index"]) // 4, int(row["theta_dot_index"]) // 4): row
                    for row in original_rows
                    if int(row["theta_index"]) % 4 == 0 and int(row["theta_dot_index"]) % 4 == 0}

    angles = np.linspace(*model.contact_angles(params), NUM_ANGLES)
    speeds = np.linspace(SETTINGS["minimum_speed"], SETTINGS["maximum_speed"], NUM_SPEEDS)
    metadata = {**parameter_metadata(params), **SETTINGS,
                "original_source_csv": source, "original_grid_rolling_percent": original_percent}
    rows = []
    for i, theta in enumerate(angles):
        for j, speed in enumerate(speeds):
            result = original.get((i, j))
            if result is None:
                result = simulate_state(theta, speed, params, SETTINGS["max_time"])
            if result["classification"] == "unclassified":
                result = simulate_state(theta, speed, params, SETTINGS["retry_time"])
            rows.append({
                "theta_index": i, "theta_dot_index": j,
                "theta0": theta, "theta_dot0": speed,
                "relative_angle": (theta - gamma) / metadata["alpha"],
                **{key: result[key] for key in (
                    "classification", "final_time", "final_theta", "final_theta_dot",
                    "total_impacts", "downhill_impacts", "uphill_impacts", "rolling_speed",
                )},
                **metadata,
            })
    write_csv(path, rows)
    return rows, path


def collect_case(params):
    """Worker task: one full parameter case, including its local return map."""
    configure_simulator()
    rows, path = collect_grid(params)
    counts = {label: sum(row["classification"] == label for row in rows)
              for label in CLASSIFICATIONS}
    fixed_point = return_map.calculate_fixed_points(params).get("rolling", np.nan)
    rolling_exists = bool(np.isfinite(fixed_point))
    multiplier, exact_multiplier, step_time = np.nan, np.nan, np.nan
    resolved = False
    if rolling_exists:
        estimates, metadata = floquet.estimate_floquet(params, timestep=SETTINGS["timestep"])
        exact_multiplier = metadata["exact_multiplier"]
        resolved = metadata["estimate_resolved"]
        if resolved:
            multiplier = estimates[-1]["multiplier"]
        result = return_map.simulate_return(fixed_point, params, timestep=SETTINGS["timestep"])
        if (result["return_type"] == "downhill_step" and result["impact_count"] == 1
                and np.isfinite(result["return_time"]) and result["return_time"] > 0):
            step_time = result["return_time"]
        floquet_path = DATA_DIR / f"summary_floquet_gamma_{params['slope_angle']:.3f}_N_{params['num_spokes']}.csv"
        write_csv(floquet_path, [{**estimate, **metadata} for estimate in estimates])
    rolling_percent = 100 * counts["rolling"] / len(rows)
    original_percent = float(rows[0]["original_grid_rolling_percent"])
    return {
        **parameter_metadata(params), **SETTINGS,
        "rolling_percent": rolling_percent,
        "settled_percent": 100 * counts["settled"] / len(rows),
        "unresolved_percent": 100 * (counts["unclassified"] + counts["numerical_failure"]) / len(rows),
        "upright_percent": 100 * counts["upright_equilibrium"] / len(rows),
        "multiplier": multiplier, "exact_multiplier": exact_multiplier,
        "rolling_exists": rolling_exists, "estimate_resolved": resolved,
        "omega_star": fixed_point, "step_time": step_time,
        "num_states": len(rows), "source_csv": path.name,
        "original_source_csv": rows[0]["original_source_csv"],
        "original_grid_rolling_percent": original_percent,
        "grid_difference_pp": rolling_percent - original_percent,
    }


def build_summary(workers=3):
    """Reuse or simulate all supported cases and save their aggregate results."""
    DATA_DIR.mkdir(exist_ok=True)
    base_params = model.generate_params()
    cases = [{**base_params, "num_spokes": spokes, "slope_angle": gamma}
             for spokes in SPOKE_COUNTS for gamma in GAMMAS if gamma < np.pi / spokes]
    print(f"Comparing {len(cases)} cases on {NUM_ANGLES} x {NUM_SPEEDS} grids.", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(collect_case, params) for params in cases]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(f"[{len(results)}/{len(cases)}] N={row['num_spokes']}, gamma={row['gamma']:.3f}: "
                  f"rolling {row['rolling_percent']:.1f}%, "
                  f"unresolved {row['unresolved_percent']:.1f}%", flush=True)
    results.sort(key=lambda row: (row["num_spokes"], row["gamma"]))
    write_csv(SUMMARY_CSV, results)
    differences = [abs(row["grid_difference_pp"]) for row in results
                   if np.isfinite(row["grid_difference_pp"])]
    if differences:
        print(f"Grid check: {len(differences)} original 81 x 161 grids; "
              f"maximum rolling-share difference = {max(differences):.2f} percentage points.", flush=True)
    print(f"Saved {SUMMARY_CSV}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    build_summary(parser.parse_args().workers)
