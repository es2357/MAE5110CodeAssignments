import numpy as np


def dynamics(t, state, params):
    position, velocity = state
    gravity = params["gravity"]

    return np.array([
        velocity,
        -gravity
    ])


def calculate_energy(state, params):
    position, velocity = state

    mass = params["mass"]
    gravity = params["gravity"]

    kinetic_energy = 0.5 * mass * velocity**2
    potential_energy = mass * gravity * position

    return kinetic_energy, potential_energy