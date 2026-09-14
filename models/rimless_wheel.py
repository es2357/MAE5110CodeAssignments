"""
RIMLESS WHEEL MODEL (assignment 1)
: defines the physical parameters, equations of motion, and collision reset
"""

from enum import Enum
import numpy as np


class ImpactDirection(Enum):
    """Contact outcome, with signed values for the wheel's reset equations"""
    DOWNHILL = 1
    UPHILL = -1
    NONE = 0


def generate_params():
    return {
        # Russ Tedrake numbers:
        # https://underactuated.csail.mit.edu/simple_legs.html#section2
        "gravity": 9.81,  # gravity m/s^2)
        "length": 1,  # rod length (m)
        "mass": 1,  # point at center
        "num_spokes": 8,  # number of spokes
        "slope_angle": 0.39,  # slope angle (rad)

        # Max angles for the model to be valid (rad)
        # For eight spokes, alpha = pi/8 = 0.3927 rad = 22.5 deg
        # For six spokes, alpha = pi/6 = 0.5236 rad = 30 deg
        # For twelve spokes, alpha = pi/12 = 0.2618 rad = 15 deg


        # 0.08 rad = 4.58 deg
        # 0.16 rad = 9.17 deg
        # 0.26 rad = 14.9 deg
        # 0.5 rad = 28.6 deg
        # 0.8 rad = 45.8 deg

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


def impact_direction(state, params) -> ImpactDirection:
    """Return the impact direction, or NONE when no contact is detected."""
    theta, theta_dot = state
    lower, upper = contact_angles(params)

    # downhill impact -- usual downhill motion
    if theta >= upper and theta_dot > 0:
        return ImpactDirection.DOWNHILL

    # uphill impact
    if theta <= lower and theta_dot < 0:
        return ImpactDirection.UPHILL

    return ImpactDirection.NONE


def reset_state(state_minus, params, direction: ImpactDirection):
    """Apply the reset to the state at the instant of impact"""

    if direction not in (ImpactDirection.DOWNHILL, ImpactDirection.UPHILL):
        raise ValueError("Impact direction must be DOWNHILL or UPHILL.")

    alpha = np.pi / params["num_spokes"]

    state_plus = np.array(state_minus, dtype=float, copy=True)

    # state reset
    state_plus[0] -= direction.value * 2 * alpha
    state_plus[1] *= np.cos(2 * alpha)

    return state_plus
