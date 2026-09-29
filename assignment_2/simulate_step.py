import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import numpy as np

from integrators import rk4 as integrator
from models import inverted_pendulum_walker as model
import stabilizing_upright as controller



def simulate_step(initial_angular_vel, angle_of_attack, theta_grid, angular_vel_grid, roa, params):
    state = np.array([0.0, initial_angular_vel], dtype=float)

    local_params = params.copy()
    local_params["angle_of_attack"] = angle_of_attack
    local_params["ankle_torque"] = 0.0


    if controller.is_valid_capture(state, theta_grid, angular_vel_grid, roa, params):
        return ("already in roa", state)

    t = 0.0
    max_time = 10.0
    timestep = 1e-3
    has_impacted = False

    while t < max_time:
        local_params["ankle_torque"] = 0.0

        next_state = integrator.integrate(
            t,
            state,
            timestep,
            model.dynamics,
            local_params
        )

        if (
            controller.roa_event_guard(state, next_state, theta_grid, angular_vel_grid, roa)
            and
            controller.is_valid_capture(next_state, theta_grid, angular_vel_grid, roa, params)
        ):
            return ("roa", next_state)

        if (not has_impacted and model.event_guard(state,next_state,local_params)):
            next_state = (model.event_dynamics(next_state,local_params))
            has_impacted = True

            if controller.is_valid_capture(next_state, theta_grid, angular_vel_grid, roa, params):
                return ("roa", next_state)
        elif (has_impacted and controller.poincare_section(state, next_state)):
            return ("poincare", next_state[1])

        state = next_state
        t += timestep

    return ("failed", np.nan)



def simulate_step_history(initial_angular_vel, angle_of_attack, theta_grid, angular_vel_grid, roa, params):
    state = np.array([0.0,initial_angular_vel],dtype=float)

    local_params = params.copy()
    local_params["angle_of_attack"] = angle_of_attack
    local_params["ankle_torque"] = 0.0

    t = 0.0
    max_time = 10.0
    timestep = 1e-3
    has_impacted = False

    time_history = [t]
    state_history = [state.copy()]

    if controller.is_valid_capture(state, theta_grid, angular_vel_grid, roa, params):
        return (
            "already in roa",
            state,
            np.asarray(time_history),
            np.asarray(state_history)
        )


    while t < max_time:
        local_params["ankle_torque"] = 0.0

        next_state = integrator.integrate(t,state, timestep, model.dynamics,local_params)
        next_t = (t + timestep)

        if (
            controller.roa_event_guard(state, next_state, theta_grid, angular_vel_grid, roa)
            and
            controller.is_valid_capture(next_state, theta_grid, angular_vel_grid, roa, params)
        ):
                time_history.append(next_t)
                state_history.append(next_state.copy())

                return (
                    "roa",
                    next_state,
                    np.asarray(time_history),
                    np.asarray(state_history)
                )


        if (not has_impacted and model.event_guard(state, next_state, local_params)):
            time_history.append(next_t)
            state_history.append(next_state.copy())

            next_state = (model.event_dynamics(next_state, local_params))
            time_history.append(next_t)

            state_history.append(next_state.copy())

            has_impacted = True

            if controller.is_valid_capture(next_state, theta_grid, angular_vel_grid, roa, params):
                return (
                    "roa",
                    next_state,
                    np.asarray(time_history),
                    np.asarray(state_history)
                )

        elif (has_impacted and controller.poincare_section(state, next_state)):
            time_history.append(next_t)
            state_history.append(next_state.copy())

            return (
                "poincare",
                next_state[1],
                np.asarray(time_history),
                np.asarray(state_history)
            )

        else:
            time_history.append(next_t)
            state_history.append(next_state.copy())

        state = next_state
        t = next_t

    return (
        "failed",
        np.nan,
        np.asarray(time_history),
        np.asarray(state_history)
    )