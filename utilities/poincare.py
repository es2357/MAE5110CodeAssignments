"""
Poincaré

helps choose landing angle to use so robot reaches RoA to balance
"""

import numpy as np
from integrators import rk4 as integrator
from utilities import roa, simulation


def event_guard(previous_state, next_state, params=None):
    """
    Return True if theta = 0 + moving forward
    -- the Poincaré section here
    """
    previous_theta = previous_state[0]
    next_theta, next_speed = next_state
    crossed_midstance = previous_theta < 0.0 <= next_theta
    moving_forward = next_speed > 0.0
    return crossed_midstance and moving_forward


def locate_event(model, guard, time, state, duration, params, event_time_tol):
    """
    Return (event time, event state) -- time and state of poincare crossing
    """
    state = np.array(state)
    event_duration = simulation.find_impact_time(
        model, time, state, duration, params,
        impact_time_tol=event_time_tol,
        guard=guard,
    )
    event_state = integrator.step(
        model.dynamics, time, state, event_duration, params
    )
    return time + event_duration, event_state


def _return_result(outcome, time, state, impact_count):
    """
    Return outcome, time, final state, impact count, and next speed or NaN
    in a dictionary
    """
    next_speed = np.nan
    if outcome == "returned":
        next_speed = float(state[1])

    return {
        "outcome": outcome,
        "time": float(time),
        "state": np.array(state, dtype=float),
        "impact_count": impact_count,
        "next_speed": next_speed,
    }


def _find_return_event(model, trajectory, params, roa_data, event_time_tol):
    """
    Return the first RoA entry or midstance return after a footstrike
    """
    times = trajectory["time_traj"]
    states = trajectory["state_traj"]
    impact_count = 0

    def roa_entry_guard(previous_state, next_state, params):
        return roa.event_guard(previous_state, next_state, roa_data)

    for index in range(1, len(times)):
        previous_time = times[index - 1]
        previous_state = states[:, index - 1]
        time = times[index]
        state = states[:, index]

        # Equal times mark an instantaneous impact reset
        impact_reset = time == previous_time
        if impact_reset:
            impact_count += 1

        if not np.all(np.isfinite(state)):
            return _return_result("failed", time, state, impact_count)

        # reset is not a midstance crossing
        reached_midstance = (
            not impact_reset
            and impact_count > 0
            and event_guard(previous_state, state)
        )

        if reached_midstance:
            time, state = locate_event(
                model, event_guard, previous_time, previous_state,
                time - previous_time, params, event_time_tol,
            )

        # Check for RoA entry up to this time
        entered_roa = roa.is_in_roa(state, roa_data)

        if entered_roa and not impact_reset:
            time, state = locate_event(
                model, roa_entry_guard, previous_time, previous_state,
                time - previous_time, params, event_time_tol,
            )
        elif reached_midstance:
            # Put the return state exactly on the Poincare section
            state = state.copy()
            state[0] = 0.0
            entered_roa = roa.is_in_roa(state, roa_data)

        if entered_roa:
            return _return_result("captured", time, state, impact_count)

        if reached_midstance:
            return _return_result("returned", time, state, impact_count)

    outcome = trajectory["outcome"]
    if outcome == "time_limit":
        outcome = "unresolved"

    return _return_result(outcome, times[-1], states[:, -1], impact_count)


def simulate_return(
    model, initial_speed, alpha, params, roa_data,
    timestep, sim_time, angle_bounds=None, event_time_tol=1e-8,
):
    """
    Test one starting speed and landing angle; return the trial result
    (return to poincaré section)
    """

    # Start at midstance
    initial_state = [0.0, initial_speed]

    if roa.is_in_roa(initial_state, roa_data):
        return _return_result("captured", 0.0, initial_state, 0)

    # Set the landing angle and leave ankle control off
    trial_params = params.copy()
    trial_params["angle_of_attack"] = alpha
    trial_params["ankle_torque"] = 0.0

    if angle_bounds is None:
        gamma = params["incline"]
        angle_bounds = (gamma - np.pi / 2, gamma + np.pi / 2)

    trajectory = simulation.simulate(
        model, initial_state, trial_params,
        timestep=timestep,
        sim_time=sim_time,
        angle_bounds=angle_bounds,
        impact_time_tol=event_time_tol,
    )

    return _find_return_event(
        model, trajectory, trial_params, roa_data, event_time_tol
    )


def build_return_table(
    model, params, roa_data, speed_values, alpha_values,
    timestep, sim_time, angle_bounds=None, event_time_tol=1e-8,
):
    """
    Test every starting speed and landing angle, and store the results
    """

    speed_values = np.array(speed_values)
    alpha_values = np.array(alpha_values)

    # Rows = starting speeds; columns = landing angles
    shape = (len(speed_values), len(alpha_values))
    next_speeds = np.full(shape, np.nan)
    outcomes = np.empty(shape, dtype=object)
    impact_counts = np.zeros(shape, dtype=int)

    for row, initial_speed in enumerate(speed_values):
        for column, alpha in enumerate(alpha_values):
            result = simulate_return(
                model, initial_speed, alpha, params, roa_data,
                timestep=timestep,
                sim_time=sim_time,
                angle_bounds=angle_bounds,
                event_time_tol=event_time_tol,
            )

            next_speeds[row, column] = result["next_speed"]
            outcomes[row, column] = result["outcome"]
            impact_counts[row, column] = result["impact_count"]

    return {
        "speed_values": speed_values,
        "alpha_values": alpha_values,
        "next_speeds": next_speeds,
        "outcomes": outcomes,
        "impact_counts": impact_counts,
    }


def _nearest_speed_index(speed, speed_values):
    """
    Return the nearest speed's index, or None for invalid/out-of-range speeds
    """
    if len(speed_values) == 0 or not np.isfinite(speed):
        return None
    if speed < np.min(speed_values) or speed > np.max(speed_values):
        return None
    return int(np.argmin(np.abs(speed_values - speed)))


def compute_step_policy(return_table):
    """
    Return the fewest footsteps and best landing angle for each speed
    """

    speed_values = return_table["speed_values"]
    alpha_values = return_table["alpha_values"]
    number_of_speeds = len(speed_values)

    minimum_steps = np.full(number_of_speeds, np.inf)
    best_alpha = np.full(number_of_speeds, np.nan)

    # Repeat so routes requiring several steps can be discovered.
    for _ in range(number_of_speeds):
        policy_changed = False

        for row in range(number_of_speeds):
            for column, alpha in enumerate(alpha_values):
                outcome = return_table["outcomes"][row, column]
                total_steps = return_table["impact_counts"][row, column]

                if outcome == "returned":
                    next_speed = return_table["next_speeds"][row, column]
                    next_index = _nearest_speed_index(next_speed, speed_values)

                    if next_index is None:
                        continue

                    # Add the steps still needed from the next speed.
                    total_steps = total_steps + minimum_steps[next_index]

                elif outcome != "captured":
                    continue

                # Save this angle if it gives a shorter route.
                if total_steps < minimum_steps[row]:
                    minimum_steps[row] = total_steps
                    best_alpha[row] = np.nan

                    if total_steps > 0:
                        best_alpha[row] = alpha

                    policy_changed = True

        if not policy_changed:
            break

    return {
        "minimum_steps": minimum_steps,
        "best_alpha": best_alpha,
    }


def compute_max_step_policy(return_table):
    """
    Return maximum_steps and best_alpha for each sampled speed.

    maximum_steps:
        finite number: longest route to the RoA
        np.inf: a repeatable cycle allows arbitrarily many steps
                before eventually reaching the RoA
        np.nan: no route to the RoA was found

    For an infinite result, best_alpha keeps the walker on a route
    towards or around a cycle.
    """
    speed_values = return_table["speed_values"]
    alpha_values = return_table["alpha_values"]
    outcomes = return_table["outcomes"]
    footstrikes = return_table["impact_counts"]
    number_of_speeds = len(speed_values)

    maximum_steps = np.full(number_of_speeds, -np.inf)
    best_alpha = np.full(number_of_speeds, np.nan)
    next_indices = np.full(outcomes.shape, -1, dtype=int)

    # Match each return speed to a sampled speed.
    for row in range(number_of_speeds):
        for column in range(len(alpha_values)):
            if outcomes[row, column] == "returned":
                next_speed = return_table["next_speeds"][row, column]
                next_index = _nearest_speed_index(next_speed, speed_values)

                if next_index is not None:
                    next_indices[row, column] = next_index

    # Extend successful routes by one transition per pass.
    # The extra pass checks for improvement caused by a cycle.
    for _ in range(number_of_speeds + 1):
        previous_steps = maximum_steps.copy()
        improved = np.zeros(number_of_speeds, dtype=bool)

        for row in range(number_of_speeds):
            for column, alpha in enumerate(alpha_values):
                outcome = outcomes[row, column]
                total_steps = footstrikes[row, column]

                if outcome == "returned":
                    next_index = next_indices[row, column]
                    if next_index < 0:
                        continue

                    total_steps += previous_steps[next_index]

                elif outcome != "captured":
                    continue

                if total_steps > maximum_steps[row]:
                    maximum_steps[row] = total_steps
                    best_alpha[row] = np.nan

                    if total_steps > 0:
                        best_alpha[row] = alpha

                    improved[row] = True

        if not np.any(improved):
            break

    # Improvement after all passes indicates a repeatable cycle
    # with a route to the RoA.
    maximum_steps[improved] = np.inf

    # States that can reach such a cycle also have no finite maximum.
    for _ in range(number_of_speeds):
        changed = False

        for row in range(number_of_speeds):
            for column, alpha in enumerate(alpha_values):
                next_index = next_indices[row, column]
                if next_index < 0:
                    continue

                if np.isposinf(maximum_steps[next_index]):
                    if not np.isposinf(maximum_steps[row]):
                        changed = True

                    maximum_steps[row] = np.inf
                    best_alpha[row] = alpha
                    break

        if not changed:
            break

    # Replace the initial placeholder for states with no known route.
    maximum_steps[np.isneginf(maximum_steps)] = np.nan

    return {
        "maximum_steps": maximum_steps,
        "best_alpha": best_alpha,
    }