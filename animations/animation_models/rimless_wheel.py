import numpy as np

# RIMLESS WHEEL MODEL


def generate_params():
    return {
        # Russ Tedrake numbers:
        # https://underactuated.csail.mit.edu/simple_legs.html#section2
        "gravity": 9.81,  # gravity m/s^2)
        "length": 1,  # rod length (m)
        "mass": 1,  # point at center
        "num_spokes": 8,  # number of spokes
        "slope_angle": 0.08,  # slope angle (rad)
    }


def contact_angles(params):
    """Return the angles at which the rimless wheel collides with the ground"""
    num_spokes = params["num_spokes"]
    slope_angle = params["slope_angle"]

    alpha = np.pi / num_spokes
    gamma = slope_angle

    # right after impact
    lower_angle = gamma - alpha
    # -->
    # --> angle changes as the wheel rotates -->
    # -->
    # next spoke impacts
    upper_angle = gamma + alpha

    return lower_angle, upper_angle


def dynamics(t, state, params):
    """State derivative during continuous stance motion."""
    theta, theta_dot = state

    gravity = params["gravity"]
    length = params["length"]

    # same as inverted pendulum dynamics -- sanity check
    theta_ddot = (gravity / length) * np.sin(theta)

    return np.array([theta_dot, theta_ddot], dtype=float)


def impact_direction(state, params):
    """
    Check whether a trial state has reached/passed a contact boundary.
         1: downhill impact
        -1: uphill impact : happens at end when w
         0: no impact
    """
    theta, theta_dot = state
    lower, upper = contact_angles(params)

    # downhill impact -- usual downhill motion
    if theta >= upper and theta_dot > 0:
        return 1

    # wheel moves uphill -- probably at the end of the simulation
    if theta <= lower and theta_dot < 0:
        return -1

    return 0


def reset_state(state_minus, params, direction):
    """Apply the reset to the state at the instant of impact"""

    if direction not in (-1, 1):
        raise ValueError("Impact direction must be +1 or -1.")

    alpha = np.pi / params["num_spokes"]

    state_plus = np.array(state_minus, dtype=float, copy=True)

    # state reset
    state_plus[0] -= direction * 2 * alpha
    state_plus[1] *= np.cos(2 * alpha)

    return state_plus
