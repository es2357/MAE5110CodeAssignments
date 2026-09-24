"""
GRID RESOLUTION

checks how fine the sim grid needs to be for reliable controller predictions

courser grids can generate incorrect footstep predictions
"""

import csv
import json
from pathlib import Path
import numpy as np
from utilities import pd_controller, poincare, roa


def _capture_count(trial):
    """Return the footstrikes before RoA entry, or NaN if it was not reached."""
    capture = trial["capture_time"]
    if capture is None:
        return np.nan

    count = 0
    for impact in trial["impacts"]:
        if impact["time"] <= capture:
            count += 1
    return count


def _event_matches(outcome, next_index, footstrikes, expected):
    """Return True if the outcome, footstrike count, and next grid row match."""
    if outcome != expected["outcome"]:
        return False
    if footstrikes != expected["footstrikes"]:
        return False
    if outcome == "returned":
        return bool(next_index == expected["next_index"])
    return True


def _trace_policy(trial, return_table, remaining_steps):
    """Return the decision comparisons and the first mismatch index (or None)."""
    speed_values = return_table["speed_values"]
    alpha_values = return_table["alpha_values"]
    trace_rows = []
    first_divergence = None
    events = trial["control_events"]
    event_index = 0

    for decision_index, decision in enumerate(trial["policy_decisions"]):
        time = float(decision["time"])
        speed = float(decision["state"][1])
        alpha = float(decision["alpha"])
        row = poincare._nearest_speed_index(speed, speed_values)
        if row is None:
            raise ValueError("A recorded policy decision is outside the return table.")
        column = int(np.argmin(np.abs(alpha_values - alpha)))
        if not np.isclose(alpha_values[column], alpha, rtol=0.0, atol=1e-12):
            raise ValueError("A recorded action is absent from the return table.")

        predicted_outcome = str(return_table["outcomes"][row, column])
        predicted_next_speed = float(return_table["next_speeds"][row, column])
        predicted_next_index = poincare._nearest_speed_index(
            predicted_next_speed, speed_values
        )
        predicted_footstrikes = int(return_table["impact_counts"][row, column])

        # Find the event after this decision.
        while event_index < len(events) and events[event_index]["time"] <= time:
            event_index += 1
        event = None
        if event_index < len(events):
            event = events[event_index]
        previous_impacts = sum(impact["time"] <= time for impact in trial["impacts"])

        if event is None:
            actual_outcome = trial["outcome"]
            if actual_outcome == "time_limit":
                actual_outcome = "unresolved"
            actual_next_speed = np.nan
            actual_next_index = None
            actual_footstrikes = len(trial["impacts"]) - previous_impacts
        else:
            if event["kind"] == "midstance":
                actual_outcome = "returned"
                actual_next_speed = float(event["state"][1])
            else:
                actual_outcome = "captured"
                actual_next_speed = np.nan
            actual_next_index = poincare._nearest_speed_index(
                actual_next_speed, speed_values
            )
            actual_footstrikes = int(event["impact_count"]) - previous_impacts

        expected = {
            "outcome": predicted_outcome,
            "next_index": predicted_next_index,
            "footstrikes": predicted_footstrikes,
        }
        diverged = not _event_matches(
            actual_outcome, actual_next_index, actual_footstrikes, expected
        )
        trace_rows.append({
            "decision": decision_index,
            "time": time,
            "actual_speed": speed,
            "grid_speed": float(speed_values[row]),
            "grid_index": row,
            "alpha": alpha,
            "predicted_remaining_steps": float(remaining_steps[row]),
            "predicted_outcome": predicted_outcome,
            "actual_outcome": actual_outcome,
            "predicted_next_speed": predicted_next_speed,
            "actual_next_speed": actual_next_speed,
            "predicted_next_index": predicted_next_index,
            "actual_next_index": actual_next_index,
            "predicted_footstrikes": predicted_footstrikes,
            "actual_footstrikes": actual_footstrikes,
            "divergence": diverged,
        })
        if diverged and first_divergence is None:
            first_divergence = decision_index

    return trace_rows, first_divergence


def _replay_transition(
    model, params, roa_data, return_table, differing,
    timestep, trial_time, event_time_tol,
):
    """Return replays from actual and sampled speeds at two timesteps."""
    speed_values = return_table["speed_values"]
    reruns = []
    table_event = {
        "outcome": differing["predicted_outcome"],
        "next_index": differing["predicted_next_index"],
        "footstrikes": differing["predicted_footstrikes"],
    }
    walking_event = {
        "outcome": differing["actual_outcome"],
        "next_index": differing["actual_next_index"],
        "footstrikes": differing["actual_footstrikes"],
    }
    starts = [("actual", differing["actual_speed"]), ("grid", differing["grid_speed"])]
    for start_kind, start_speed in starts:
        for replay_timestep in (timestep, timestep / 2):
            result = poincare.simulate_return(
                model, start_speed, differing["alpha"],
                params, roa_data, timestep=replay_timestep,
                sim_time=trial_time, event_time_tol=event_time_tol,
            )
            next_index = poincare._nearest_speed_index(
                result["next_speed"], speed_values
            )
            reruns.append({
                "start": start_kind,
                "initial_speed": start_speed,
                "timestep": float(replay_timestep),
                "outcome": result["outcome"],
                "time": float(result["time"]),
                "next_speed": float(result["next_speed"]),
                "next_index": next_index,
                "footstrikes": int(result["impact_count"]),
                "matches_table": _event_matches(
                    result["outcome"], next_index, result["impact_count"], table_event
                ),
                "matches_walking": _event_matches(
                    result["outcome"], next_index, result["impact_count"], walking_event
                ),
            })

    return reruns


def diagnose_policy(
    model, params, simulate_walking, roa_data, return_table, step_policy,
    initial_speed, timestep, trial_time=10.0, event_time_tol=1e-8,
):
    """Return the decision trace and a summary of predictions, walking, and replays."""
    speed_values = return_table["speed_values"]
    initial_index = poincare._nearest_speed_index(initial_speed, speed_values)
    if initial_index is None:
        raise ValueError("The diagnostic speed must be inside the return table.")
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError("The diagnostic timestep must be positive and finite.")
    if not np.isfinite(trial_time) or trial_time <= 0:
        raise ValueError("The diagnostic duration must be positive and finite.")
    if not np.isfinite(event_time_tol) or event_time_tol <= 0:
        raise ValueError("The event time tolerance must be positive and finite.")

    if "minimum_steps" in step_policy:
        count_key = "minimum_steps"
        policy_name = "minimum"
    else:
        count_key = "maximum_steps"
        policy_name = "maximum"
    trial = simulate_walking(
        model, [0.0, initial_speed], params, roa_data, return_table, step_policy,
        timestep=timestep, sim_time=trial_time, event_time_tol=event_time_tol,
    )
    if "control_events" not in trial:
        raise ValueError("The walking simulator must record control_events for diagnosis.")

    trace_rows, first_divergence = _trace_policy(
        trial, return_table, step_policy[count_key]
    )

    balanced = bool(roa.has_balanced(trial))
    capture_count = _capture_count(trial)
    predicted_steps = float(step_policy[count_key][initial_index])
    summary = {
        "policy": policy_name,
        "initial_speed": float(initial_speed),
        "timestep": float(timestep),
        "predicted_steps": predicted_steps,
        "capture_count": capture_count,
        "observed_steps": capture_count if balanced else np.nan,
        "capture_time": trial["capture_time"],
        "balanced": balanced,
        "matches": bool(
            balanced and np.isfinite(predicted_steps)
            and np.isfinite(capture_count) and predicted_steps == capture_count
        ),
        "outcome": trial["outcome"],
        "first_divergence_index": first_divergence,
        "reruns": [],
    }

    if first_divergence is not None:
        summary["reruns"] = _replay_transition(
            model, params, roa_data, return_table, trace_rows[first_divergence],
            timestep, trial_time, event_time_tol,
        )

    return trace_rows, summary


def _select_grid(results, roa_change_tol, compare_membership=True):
    """Return the coarsest passing grid's index, or None if none qualifies."""
    # The last grid has no finer comparison.
    for index, candidate in enumerate(results[:-1]):
        if not np.all(candidate["matches"]):
            continue

        passes = True
        for finer in results[index + 1:]:
            same_counts = np.array_equal(candidate["observed"], finer["observed"])
            if not np.all(finer["matches"]) or not same_counts:
                passes = False
                break

            if compare_membership:
                roa_change = np.mean(candidate["membership"] != finer["membership"])
                if roa_change > roa_change_tol:
                    passes = False
                    break

        if passes:
            return index

    return None


def _read_speed_settings(directory):
    """Return the settings for a saved study that varies only velocity resolution."""
    path = directory / "settings.json"
    settings = json.loads(path.read_text())
    required = {"study", "grids", "test_speeds", "params", "timestep", "trial_time"}
    if not required.issubset(settings):
        raise ValueError(f"{path}: required study settings are missing.")
    if settings["study"] != "speed":
        raise ValueError(f"{path}: only velocity-resolution studies can be combined.")
    speeds = settings["test_speeds"]
    if not speeds or len(set(speeds)) != len(speeds):
        raise ValueError(f"{path}: test speeds must be nonempty and distinct.")
    settings["test_speeds"] = sorted(speeds)
    fixed_sizes = {(grid[0], grid[1], grid[3]) for grid in settings["grids"]}
    if len(fixed_sizes) != 1:
        raise ValueError(f"{path}: RoA and action sample counts must remain fixed.")
    return settings


def _read_speed_trials(directory, settings):
    """Return complete trial arrays for the grids declared in a saved study."""
    path = directory / "trials.csv"
    grid_columns = ("n_theta", "n_theta_dot", "n_speed", "n_alpha")
    required = set(grid_columns) | {
        "policy", "initial_speed", "predicted_steps", "observed_steps", "outcome", "balanced",
    }
    grouped = {}
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required trial columns are missing.")
        for record in reader:
            grid = tuple(int(record[name]) for name in grid_columns)
            grouped.setdefault(grid, []).append(record)
    if set(grouped) != {tuple(grid) for grid in settings["grids"]}:
        raise ValueError(f"{path}: trial grids do not match the saved settings.")

    speeds = np.array(settings["test_speeds"])
    policies = ("minimum", "maximum")
    expected = [(policy, speed) for policy in policies for speed in speeds]
    results = {}
    for grid, records in grouped.items():
        ordered = []
        for policy in policies:
            rows = [record for record in records if record["policy"] == policy]
            ordered.extend(sorted(rows, key=lambda record: float(record["initial_speed"])))
        actual = [(record["policy"], float(record["initial_speed"])) for record in ordered]
        if actual != expected or len(ordered) != len(records):
            raise ValueError(f"{path}: grid {grid} has missing, duplicate, or unexpected trials.")
        predicted = np.array([float(row["predicted_steps"]) for row in ordered]).reshape(2, -1)
        observed = np.array([float(row["observed_steps"]) for row in ordered]).reshape(2, -1)
        balanced = np.array([row["balanced"] == "True" for row in ordered]).reshape(2, -1)
        results[grid] = {
            "grid": grid, "test_speeds": speeds.copy(),
            "predicted": predicted, "observed": observed, "balanced": balanced,
            "outcomes": np.array([row["outcome"] for row in ordered]).reshape(2, -1),
            "matches": balanced & np.isfinite(predicted) & np.isfinite(observed) & (predicted == observed),
        }
    return results


def load_speed_studies(study_directories):
    """Return merged velocity-study results, shared settings, and the selected grid index."""
    combined = {}
    shared = None
    compatible_keys = ("params", "timestep", "trial_time", "test_speeds")
    for directory in study_directories:
        directory = Path(directory)
        settings = _read_speed_settings(directory)
        if shared is None:
            shared = settings.copy()
        else:
            for key in compatible_keys:
                if settings[key] != shared[key]:
                    raise ValueError(f"{directory}: {key} differs between the saved studies.")
            first_grid = shared["grids"][0]
            current_grid = settings["grids"][0]
            if any(first_grid[index] != current_grid[index] for index in (0, 1, 3)):
                raise ValueError(f"{directory}: RoA or action resolution differs between studies.")

        for grid, result in _read_speed_trials(directory, settings).items():
            if grid in combined:
                previous = combined[grid]
                for key in ("predicted", "observed", "balanced", "outcomes"):
                    equal_nan = key in ("predicted", "observed")
                    if not np.array_equal(previous[key], result[key], equal_nan=equal_nan):
                        raise ValueError(f"{directory}: conflicting results for repeated grid {grid}.")
            else:
                combined[grid] = result

    if shared is None:
        raise ValueError("Provide at least one saved velocity-study directory.")
    results = sorted(combined.values(), key=lambda result: result["grid"][2])
    # These studies have identical RoA settings; only velocity resolution varies.
    selected = _select_grid(results, roa_change_tol=0.0, compare_membership=False)
    shared["grids"] = [list(result["grid"]) for result in results]
    shared["selected_grid"] = None if selected is None else list(results[selected]["grid"])
    return results, shared, selected


def check_grid_resolution(
    model, params, simulate_walking, grid_sizes, test_speeds,
    balance_timestep, table_timestep, walking_timestep,
    roa_change_tol=0.01, trial_time=10.0,
):
    """Return grid results, trial details, and the selected grid index (or None)."""
    if (
        not np.isfinite(params["gravity"]) or params["gravity"] <= 0
        or not np.isfinite(params["length"]) or params["length"] <= 0
    ):
        raise ValueError("Gravity and leg length must be positive and finite.")
    maximum_speed = np.sqrt(2 * params["gravity"] / params["length"])
    test_speeds = np.asarray(test_speeds, dtype=float)
    if (
        test_speeds.ndim != 1 or test_speeds.size == 0
        or not np.all(np.isfinite(test_speeds))
        or np.any(test_speeds < 0) or np.any(test_speeds > maximum_speed)
    ):
        raise ValueError("Test speeds must be a nonempty finite list inside the speed range.")
    if not np.isfinite(roa_change_tol) or not 0 <= roa_change_tol <= 1:
        raise ValueError("RoA disagreement tolerance must lie between zero and one.")
    for value in (balance_timestep, table_timestep, walking_timestep, trial_time):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Timesteps and trial duration must be positive and finite.")

    grids = []
    for grid in grid_sizes:
        if len(grid) != 4 or any(
            not isinstance(size, (int, np.integer)) or isinstance(size, (bool, np.bool_))
            or size < 2 for size in grid
        ):
            raise ValueError("Each grid must contain four integer sample counts of at least two.")
        grid = tuple(int(size) for size in grid)
        if grids and (
            any(size < previous for size, previous in zip(grid, grids[-1]))
            or grid == grids[-1]
        ):
            raise ValueError("Each grid must refine at least one dimension without coarsening another.")
        grids.append(grid)
    if not grids:
        raise ValueError("Provide at least one grid configuration.")

    balance_params = params.copy()
    balance_params["angle_of_attack"] = np.pi / 8

    bounds = (
        params["incline"] - np.pi / 2,
        params["incline"] + np.pi / 8,
    )

    # Use the same RoA probe states for every grid.
    probe_states = []
    for speed in np.linspace(-0.4, 0.4, 61):
        for angle in np.linspace(-0.1, 0.1, 81):
            probe_states.append([angle, speed])

    results = []
    details = []
    roa_cache = {}

    for grid in grids:
        n_theta, n_theta_dot, n_speed, n_alpha = grid
        print("Testing grid:", grid, flush=True)

        # Reuse RoA estimates with the same size.
        roa_size = (n_theta, n_theta_dot)

        if roa_size not in roa_cache:
            print(f"Estimating RoA on {n_theta} x {n_theta_dot} samples...", flush=True)
            roa_cache[roa_size] = roa.estimate_roa(
                model,
                pd_controller.compute_ankle_torque,
                balance_params,
                np.linspace(-0.1, 0.1, n_theta),
                np.linspace(-0.4, 0.4, n_theta_dot),
                timestep=balance_timestep,
                sim_time=5.0,
                angle_bounds=bounds,
            )

        roa_data = roa_cache[roa_size]

        membership = []
        for state in probe_states:
            membership.append(roa.is_in_roa(state, roa_data))
        membership = np.array(membership)

        print(f"Building {n_speed} x {n_alpha} return table...", flush=True)
        table = poincare.build_return_table(
            model, params, roa_data,
            np.linspace(0.0, maximum_speed, n_speed),
            np.linspace(np.pi / 8, np.pi / 7, n_alpha),
            timestep=table_timestep,
            sim_time=5.0,
        )

        policies = [
            ("minimum", poincare.compute_step_policy(table), "minimum_steps"),
            ("maximum", poincare.compute_max_step_policy(table), "maximum_steps"),
        ]

        predicted = np.full((2, len(test_speeds)), np.nan)
        observed = np.full_like(predicted, np.nan)
        capture_counts = np.full_like(predicted, np.nan)
        balanced_trials = np.zeros(predicted.shape, dtype=bool)

        print(f"Validating {predicted.size} policy trials...", flush=True)
        for row, (name, policy, count_key) in enumerate(policies):
            for column, speed in enumerate(test_speeds):
                index = poincare._nearest_speed_index(
                    speed, table["speed_values"]
                )

                if index is not None:
                    predicted[row, column] = policy[count_key][index]

                trial = simulate_walking(
                    model, [0.0, speed], params, roa_data, table, policy,
                    timestep=walking_timestep,
                    sim_time=trial_time,
                )

                balanced = roa.has_balanced(trial)
                balanced_trials[row, column] = balanced
                capture_counts[row, column] = _capture_count(trial)

                # Verify counts only after balancing.
                if balanced:
                    observed[row, column] = capture_counts[row, column]

                details.append([
                    *grid, name, speed,
                    predicted[row, column],
                    observed[row, column],
                    trial["outcome"],
                    balanced,
                ])

        matches = (
            np.isfinite(predicted)
            & np.isfinite(observed)
            & (predicted == observed)
        )

        per_policy = {}
        for row, (name, _, _) in enumerate(policies):
            per_policy[name] = {
                "tested": len(test_speeds),
                "captured": int(np.isfinite(capture_counts[row]).sum()),
                "balanced": int(balanced_trials[row].sum()),
                "matched": int(matches[row].sum()),
                "nonfinite_predictions": int((~np.isfinite(predicted[row])).sum()),
            }

        results.append({
            "grid": grid,
            "test_speeds": test_speeds.copy(),
            "predicted": predicted,
            "observed": observed,
            "capture_counts": capture_counts,
            "balanced": balanced_trials,
            "matches": matches,
            "membership": membership,
            "per_policy": per_policy,
        })

        print(
            f"Matched and balanced: {matches.sum()}/{matches.size}",
            flush=True,
        )

    selected = _select_grid(results, roa_change_tol)
    return results, details, selected
