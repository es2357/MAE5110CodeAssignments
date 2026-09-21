import numpy as np
from utilities import pd_controller, poincare, roa


def check_grid_resolution(
    model, params, simulate_walking, grid_sizes, test_speeds,
    balance_timestep, table_timestep, walking_timestep,
    roa_change_tol=0.01, trial_time=10.0,
):
    """Compare grids using the same test speeds and RoA probe states."""
    maximum_speed = np.sqrt(2 * params["gravity"] / params["length"])

    balance_params = params.copy()
    balance_params["angle_of_attack"] = np.pi / 8

    bounds = (
        params["incline"] - np.pi / 2,
        params["incline"] + np.pi / 8,
    )

    # Compare every RoA estimate at the same states.
    probe_states = []
    for speed in np.linspace(-0.4, 0.4, 61):
        for angle in np.linspace(-0.1, 0.1, 81):
            probe_states.append([angle, speed])

    results = []
    details = []
    roa_cache = {}

    for grid in grid_sizes:
        n_theta, n_theta_dot, n_speed, n_alpha = grid
        print("Testing grid:", grid, flush=True)

        # Reuse the RoA when only the return-table resolution changes.
        roa_size = (n_theta, n_theta_dot)

        if roa_size not in roa_cache:
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

                capture = trial["capture_time"]
                balanced = roa.has_balanced(trial)

                # Count a trial as verified only if it actually balances.
                if capture is not None and balanced:
                    count = 0
                    for impact in trial["impacts"]:
                        if impact["time"] <= capture:
                            count += 1

                    observed[row, column] = count

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

        results.append({
            "grid": grid,
            "predicted": predicted,
            "observed": observed,
            "matches": matches,
            "membership": membership,
        })

        print(
            f"Matched and balanced: {matches.sum()}/{matches.size}",
            flush=True,
        )

    # A candidate must agree with every finer grid tested.
    # The finest grid cannot qualify without a further comparison.
    selected = None

    for index in range(len(results) - 1):
        candidate = results[index]
        passes = bool(np.all(candidate["matches"]))

        for finer in results[index + 1:]:
            same_counts = np.array_equal(
                candidate["observed"], finer["observed"]
            )

            roa_change = np.mean(
                candidate["membership"] != finer["membership"]
            )

            if (
                not np.all(finer["matches"])
                or not same_counts
                or roa_change > roa_change_tol
            ):
                passes = False

        if passes:
            selected = index
            break

    return results, details, selected