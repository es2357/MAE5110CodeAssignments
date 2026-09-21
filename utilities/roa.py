"""
ROA

estimates controller's RoA and uses the estimate during walking
"""

import numpy as np
from utilities import simulation


def _near_upright(state, angle_tol, speed_tol):
    """
    Returns True when absolute angle and angular speed are below
    their tolerances -- checks if robot is almost upright and slow
    """
    theta, theta_dot = state
    return (abs(theta) < angle_tol) & (abs(theta_dot) < speed_tol)


def _neighbor_indices(value, grid_values):
    """
    Return the matching index or the two indices on either side
    grid values = theta_values or theta_dot_values

    Used by is_in_roa to find the surrounding sampled states
    """

    # The value must be within the grid
    if value < grid_values[0] or value > grid_values[-1]:
        return []

    for index, grid_value in enumerate(grid_values):
        # The value lands exactly on a grid point
        if value == grid_value:
            return [index]

        # The first larger grid point is the right neighbor
        # The previous grid point is the left neighbor
        if grid_value > value:
            left_index = index - 1
            right_index = index
            return [left_index, right_index]


def has_balanced(result, angle_tol=1e-3, speed_tol=1e-3, settle_time=0.5):
    """
    Return True if the robot stayed near upright "long enough"
    """

    if result["outcome"] != "time_limit":
        return False

    times = result["time_traj"]
    near_upright = _near_upright(
        result["state_traj"], angle_tol, speed_tol
    )
    end_time = times[-1]

    # Allow tiny margin for floating-point rounding
    roundoff = 4 * np.finfo(float).eps * max(end_time, settle_time)

    # Check backward from the final recorded state
    for time, is_near in zip(reversed(times), reversed(near_upright)):
        if not is_near:
            return False

        if end_time - time >= settle_time - roundoff:
            return True

    return False


def estimate_roa(
    model, controller, params, theta_values, theta_dot_values,
    timestep, sim_time, angle_bounds,
    angle_tol=1e-3, speed_tol=1e-3, settle_time=0.5,
):
    """
    Return the sampled angles, speeds, and a grid of balancing results
    """

    theta_values = np.array(theta_values)
    theta_dot_values = np.array(theta_dot_values)

    # Rows = speeds; columns = angles
    grid_shape = (len(theta_dot_values), len(theta_values))
    converged = np.zeros(grid_shape, dtype=bool)

    for row, theta_dot in enumerate(theta_dot_values):
        for column, theta in enumerate(theta_values):
            result = simulation.simulate(
                model, [theta, theta_dot], params,
                timestep=timestep,
                sim_time=sim_time,
                controller=controller,
                angle_bounds=angle_bounds,
                stop_on_impact=True,
            )

            converged[row, column] = has_balanced(
                result,
                angle_tol=angle_tol,
                speed_tol=speed_tol,
                settle_time=settle_time,
            )

    return {
        "theta_values": theta_values,
        "theta_dot_values": theta_dot_values,
        "converged": converged,
    }


def is_in_roa(state, roa_data):
    """
    Return True if all surrounding grid points successfully balanced
    -- checks if state given is in RoA
    """

    if not np.all(np.isfinite(state)):
        return False

    theta, theta_dot = state

    angle_indices = _neighbor_indices(theta, roa_data["theta_values"])
    speed_indices = _neighbor_indices(theta_dot, roa_data["theta_dot_values"])

    # The state must be within the sampled grid
    if not angle_indices or not speed_indices:
        return False

    converged = roa_data["converged"]

    # Every surrounding grid point must have balanced
    for speed_index in speed_indices:
        for angle_index in angle_indices:
            if not converged[speed_index, angle_index]:
                return False

    return True


# checks whether robot moved outside of RoA
def event_guard(previous_state, next_state, roa_data):
    """
    Detect an outside-to-inside transition at the sampled states
    Check is_in_roa(initial_state, roa_data) separately at startup.
    """
    was_inside = is_in_roa(previous_state, roa_data)
    is_inside = is_in_roa(next_state, roa_data)
    return not was_inside and is_inside
