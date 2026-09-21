"""
PD Controller (assignment 2)
"""
import numpy as np


def compute_ankle_torque(state, params):
    """Gravity cancellation + PD stabilization, with torque saturation."""
    theta, theta_dot = state

    m = params["mass"]
    length = params["length"]
    g = params["gravity"]
    kp = params["balance_kp"]
    kd = params["balance_kd"]

    desired_acceleration = -kp * theta - kd * theta_dot

    raw_torque = m * length**2 * (
        desired_acceleration - (g / length) * np.sin(theta)
    )

    torque_min = -0.1 * m * g * length
    torque_max = 0.05 * m * g * length

    return float(np.clip(raw_torque, torque_min, torque_max))
