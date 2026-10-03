
"""
BOUNCING BALL MODEL (assignment 0)
"""

import numpy as np

def generate_params():
    return {
        "gravity": 9.81,
        "mass": 1.0,
        "restitution": 0.8,
    }


def generate_initial_condition():
    return np.array([1.0, 0.0])


def dynamics(t, state, params):
    """Return the state derivative while the ball is in flight."""

    height, velocity = state
    gravity = params["gravity"]

    return np.array([velocity, -gravity])


def resolve_impact(state, params):
    """Prevent ground penetration and reverse downward velocity."""
    height, velocity = state
    restitution = params["restitution"]

    corrected_state = state.copy()

    if height <= 0 and velocity < 0:
        corrected_state[0] = 0
        corrected_state[1] = -restitution * velocity

    return corrected_state


def calculate_energy(state, params):
    mass = params["mass"]
    gravity = params["gravity"]

    height = state[0]
    velocity = state[1]

    kinetic_energy = 0.5 * mass * velocity**2
    potential_energy = mass * gravity * height

    return kinetic_energy, potential_energy