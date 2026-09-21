"""
Simulation

Time stepping, impact timing, and individual simulations
"""

import numpy as np
from integrators import rk4 as integrator


def find_impact_time(
    model, t, state, timestep, params, impact_time_tol=1e-8, guard=None,
):
    """
    Return the time until a known crossing using bisection.
    Use the model's impact guard unless another guard is provided.
    """
    if guard is None:
        guard = model.event_guard

    # The crossing occurs somewhere in between
    t_minus = 0.0
    t_plus = timestep

    # Keep narrowing the interval
    while t_plus - t_minus > impact_time_tol:
        t_mid = 0.5 * (t_minus + t_plus)

        state_mid = integrator.step(
            model.dynamics, t, state, t_mid, params
        )
        crossed = guard(state, state_mid, params)
        if crossed:
            # The crossing occurred by the midpoint: search the first half
            t_plus = t_mid
        else:
            # The crossing has not occurred yet: search the second half
            t_minus = t_mid

    return t_plus


def _step_to_impact(model, t, state, timestep, params, impact_time_tol):
    """
    Return (state, elapsed time, impact detected), before reset
    -- move sim forward one timestep, stopping at detected impact
    """
    next_state = integrator.step(model.dynamics, t, state, timestep, params)

    if not np.all(np.isfinite(next_state)):
        return next_state, timestep, False
    if not model.event_guard(state, next_state, params):
        return next_state, timestep, False

    duration = find_impact_time(
        model, t, state, timestep, params,
        impact_time_tol=impact_time_tol,
    )
    state_minus = integrator.step(model.dynamics, t, state, duration, params)
    return state_minus, duration, True


def _check_state(state, angle_bounds):
    """
    Return "failed" for an invalid state, or None if it is valid
    """
    if not np.all(np.isfinite(state)):
        return "failed"

    if angle_bounds is not None:
        angle_min, angle_max = angle_bounds
        if not angle_min < state[0] < angle_max:
            return "failed"

    return None


def advance_timestep(
    model, t, state, timestep, params, *,
    angle_bounds=None, stop_on_impact=False, impact_time_tol=1e-8,
):
    """
    Advance one timestep with fixed controls and handle impacts

    Return dictionary containing:
        time_traj: sample times, including the starting time t.
        state_traj: states as columns, including the starting state.
        impacts: records with time, state_minus, and state_plus.
        outcome: "completed", "impact", or "failed"
    """
    current_state = np.array(state, dtype=float)
    current_time = t
    end_time = t + timestep
    times = [current_time]
    states = [current_state.copy()]
    impacts = []
    outcome = _check_state(current_state, angle_bounds)

    while current_time < end_time and outcome is None:
        current_state, duration, detect_impact = _step_to_impact(
            model, current_time, current_state, end_time - current_time, params,
            impact_time_tol,
        )
        if detect_impact:
            current_time = min(current_time + duration, end_time)
        else:
            current_time = end_time
        times.append(current_time)
        states.append(current_state.copy())

        if not np.all(np.isfinite(current_state)):
            outcome = "failed"
            break

        if detect_impact:
            impact = {
                "time": current_time,
                "state_minus": current_state.copy(),
                "state_plus": None,
            }
            impacts.append(impact)
            if stop_on_impact:
                outcome = "impact"
                break

            current_state = model.event_dynamics(current_state.copy(), params)
            impact["state_plus"] = current_state.copy()
            times.append(current_time)
            states.append(current_state.copy())

        outcome = _check_state(current_state, angle_bounds)

    return {
        "time_traj": np.array(times),
        "state_traj": np.array(states).T,
        "impacts": impacts,
        "outcome": "completed" if outcome is None else outcome,
    }


def impact_step(model, t, state, timestep, params, impact_time_tol=1e-8):
    """
    Advance one timestep with resets; return the final state and impacts
    """
    result = advance_timestep(
        model, t, state, timestep, params,
        impact_time_tol=impact_time_tol,
    )
    return result["state_traj"][:, -1], result["impacts"]


def simulate(
    model, initial_state, params, timestep, sim_time,
    controller=None, angle_bounds=None, stop_on_impact=False,
    impact_time_tol=1e-8,
):
    """
    Run a simulation and return its recorded motion, impacts, and outcome
    """

    state = np.array(initial_state, dtype=float)
    trial_params = params.copy()
    time = 0.0

    times = [time]
    states = [state.copy()]
    impacts = []
    outcome = _check_state(state, angle_bounds)

    while time < sim_time and outcome is None:
        # Shorten the last step if needed
        next_time = min(time + timestep, sim_time)
        step_duration = next_time - time

        # Update the ankle torque if a controller is provided.
        if controller is not None:
            torque = controller(state, trial_params)

            if not np.isfinite(torque):
                outcome = "failed"
                break

            trial_params["ankle_torque"] = torque

        # Advance sim
        result = advance_timestep(
            model, time, state, step_duration, trial_params,
            angle_bounds=angle_bounds,
            stop_on_impact=stop_on_impact,
            impact_time_tol=impact_time_tol,
        )

        time = result["time_traj"][-1]
        state = result["state_traj"][:, -1]

        # Save new samples, skipping the starting point already recorded
        times.extend(result["time_traj"][1:])
        states.extend(result["state_traj"][:, 1:].T)
        impacts.extend(result["impacts"])

        # A stopping outcome prevents another loop iteration
        if result["outcome"] != "completed":
            outcome = result["outcome"]

    if outcome is None:
        outcome = "time_limit"

    return {
        "time_traj": np.array(times),
        "state_traj": np.array(states).T,
        "impacts": impacts,
        "outcome": outcome,
    }
