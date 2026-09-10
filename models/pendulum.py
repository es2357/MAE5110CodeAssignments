import numpy as np

# PENDULUM MODEL

def dynamics(t, state, params):
    # params : dictionary containing the parameters of the pendulum
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]
    damping_coeff = params["damping_coeff"]

    angle = state[0]
    angular_velocity = state[1]

    # can change equations to make PE = 0 and theta = 0 at
    # the bottom of the swing: negate angular_acceleration
    # and use (1 - cos(angle))

    angular_acceleration = (
        mass * gravity * length * np.sin(angle)
        - damping_coeff * angular_velocity  # <-- DAMPING TERM
    ) / (mass * length**2)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative


def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "length": 1,  # rod length (m)
        "mass": 1,  # point mass at end of rod (kg)
        "damping_coeff": 0.1,  # damping coefficient (kg*m^2/s)
    }
    return params


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]  # indexes entire row "vectorized" if state is (2, N)
    angular_velocity = state[1]

    kinetic_energy = 0.5 * mass * (length * angular_velocity) ** 2
    # potential_energy = mass * gravity * length * (1 - np.cos(angle))
    potential_energy = mass * gravity * length * (np.cos(angle))
    return kinetic_energy, potential_energy
