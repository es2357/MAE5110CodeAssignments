import numpy as np

from models import pendulum
from integrators import rk4


def test_pendulum_energy_conservation():
    """
    Test that energy ramains constant when
    damping = 0 and torque = 0
    """
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 0.0
    state = np.array([0.2, 0.0])
    timestep = 0.01
    kinetic, potential = pendulum.calculate_energy(state, params)
    initial_energy = kinetic + potential

    for step in range(100):
        time = step * timestep
        state = rk4(pendulum.dynamics, time, state, timestep, params)
        kinetic, potential = pendulum.calculate_energy(state, params)
        delta_energy = kinetic + potential - initial_energy

        assert np.isclose(delta_energy, 0.0, atol=1e-6, rtol=0.0)


def test_torque():
    """
    Test that ang_vel predicted by physics
    matches simulated ang_vel to test torque
    """
    params = pendulum.generate_params()
    params["gravity"] = 0.0
    params["damping_coeff"] = 0.0
    params["torque"] = 1.0
    state = np.array([0.2, 0.0])
    timestep = 0.01

    for step in range(100):
        time = step * timestep
        state = rk4(pendulum.dynamics, time, state, timestep, params)

    duration = 100 * timestep
    inertia = params["mass"] * params["length"] ** 2
    expected_vel = params["torque"] * duration / inertia
    assert np.isclose(state[1], expected_vel, atol=1e-6, rtol=0.0)


def test_damping():
    """
    Test that damping slowed down pendulum as expected
    """
    params = pendulum.generate_params()
    params["gravity"] = 0.0
    params["torque"] = 0.0
    params["damping_coeff"] = 0.5
    state = np.array([0.2, 1.0])
    omega_0 = state[1]
    timestep = 0.01

    for step in range(100):
        time = step * timestep
        state = rk4(pendulum.dynamics, time, state, timestep, params)

    duration = 100 * timestep
    inertia = params["mass"] * params["length"] ** 2
    damping = params["damping_coeff"]
    expected_vel = omega_0 * np.exp(-damping * duration / inertia)

    assert np.isclose(state[1], expected_vel, atol=1e-6, rtol=0.0)
