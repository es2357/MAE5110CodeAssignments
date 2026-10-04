import numpy as np

from integrators import rk4
from models import pendulum


def energy_trajectory(params):
    """Record total energy initially and after each of 100 RK4 steps."""
    state = np.array([0.3, 0.5])
    timestep = 0.001
    energies = []

    for step in range(101):
        kinetic, potential = pendulum.calculate_energy(state, params)
        energies.append(kinetic + potential)

        if step < 100:
            state = rk4(
                pendulum.dynamics,
                step * timestep,
                state,
                timestep,
                params,
            )

    return np.array(energies)


def test_energy_conservation():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 0.0

    energies = energy_trajectory(params)

    assert np.all(
        np.isclose(energies, energies[0], rtol=0.0, atol=1e-8)
    )


def test_torque():
    params = pendulum.generate_params()
    params["mass"] = 2.0
    params["length"] = 0.7
    params["damping_coeff"] = 0.0
    state = np.array([0.0, 0.0])

    for torque in [-0.6, 0.6]:
        params["torque"] = torque
        acceleration = pendulum.dynamics(0.0, state, params)[1]
        expected = torque / (params["mass"] * params["length"]**2)

        assert np.isclose(acceleration, expected)
    # With gravity and damping disabled, constant torque gives
    # constant angular acceleration.
    params["gravity"] = 0.0
    timestep = 0.001
    steps = 100

    for torque in [-0.6, 0.6]:
        params["torque"] = torque
        state = np.array([0.0, 0.0])

        for k in range(steps):
            state = rk4(
                pendulum.dynamics, k * timestep, state, timestep, params
            )

        expected_velocity = (
            torque / (params["mass"] * params["length"]**2)
            * steps * timestep
        )
        assert np.isclose(state[1], expected_velocity)

def test_damping():
    params = pendulum.generate_params()
    params["mass"] = 2.0
    params["length"] = 0.7
    params["torque"] = 0.0
    params["damping_coeff"] = 0.2

    for velocity in [-0.5, 0.5]:
        state = np.array([0.0, velocity])
        acceleration = pendulum.dynamics(0.0, state, params)[1]
        expected = (
            -params["damping_coeff"] * velocity
            / (params["mass"] * params["length"]**2)
        )

        assert np.isclose(acceleration, expected)

    energies = energy_trajectory(params)
    assert np.all(np.diff(energies) <= 1e-10)
    assert energies[-1] < energies[0] - 1e-6