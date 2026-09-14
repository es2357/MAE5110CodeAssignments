"""Physical invariants of the event-resolved rimless wheel simulation."""

import numpy as np
import pytest

from animation_models.rimless_wheel import contact_angles, generate_params
from animation_models.rimless_wheel_energy import simulate


@pytest.fixture(scope="module")
def rolling_stances():
    return simulate()


def test_continuous_motion_conserves_energy_and_stays_on_contour(rolling_stances):
    params = generate_params()
    lower, upper = contact_angles(params)
    for stance in rolling_stances:
        assert stance.time.size == 64
        assert np.all(np.diff(stance.time) > 0)
        assert np.all(np.diff(stance.theta) > 0)
        np.testing.assert_allclose(stance.theta[[0, -1]], [lower, upper], atol=1e-11)
        np.testing.assert_allclose(stance.energy, stance.energy[0], atol=2e-9, rtol=0)
        np.testing.assert_allclose(
            stance.local_energy, stance.local_energy[0], atol=2e-9, rtol=0
        )
        np.testing.assert_allclose(stance.energy, stance.kinetic + stance.potential)
        # Each phase trajectory lies on omega(theta) from its energy contour.
        expected_omega = np.sqrt(
            2
            * (
                stance.local_energy[0]
                - params["mass"]
                * params["gravity"]
                * params["length"]
                * np.cos(stance.theta)
            )
            / (params["mass"] * params["length"] ** 2)
        )
        np.testing.assert_allclose(stance.omega, expected_omega, atol=2e-9, rtol=0)


def test_impact_preserves_hub_position_and_loses_only_kinetic_energy(rolling_stances):
    params = generate_params()
    length, mass, gravity = (params[key] for key in ("length", "mass", "gravity"))
    alpha = np.pi / params["num_spokes"]
    restitution = np.cos(2 * alpha)
    for stance in rolling_stances:
        hub_before = stance.foot + length * np.array(
            [np.sin(stance.theta[-1]), np.cos(stance.theta[-1])]
        )
        hub_after = stance.after_foot + length * np.array(
            [np.sin(stance.after_state[0]), np.cos(stance.after_state[0])]
        )
        np.testing.assert_allclose(hub_after, hub_before, atol=1e-12)
        assert stance.after_potential == pytest.approx(stance.potential[-1], abs=1e-12)
        assert stance.after_kinetic == pytest.approx(
            stance.kinetic[-1] * restitution**2
        )
        assert stance.loss == pytest.approx(stance.kinetic[-1] * np.sin(2 * alpha) ** 2)
        assert stance.after_energy == pytest.approx(stance.energy[-1] - stance.loss)
        assert stance.loss > 0
        # Local phase-plane H has a changed foot datum at reset; world energy
        # must account for that datum instead of implying a potential jump.
        after_local_energy = stance.after_kinetic + mass * gravity * length * np.cos(
            stance.after_state[0]
        )
        assert after_local_energy + mass * gravity * stance.after_foot[
            1
        ] == pytest.approx(stance.after_energy)


def test_stances_join_at_reset_and_satisfy_analytic_return_map(rolling_stances):
    params = generate_params()
    alpha, gamma = np.pi / params["num_spokes"], params["slope_angle"]
    accumulated_loss = 0.0
    for index, stance in enumerate(rolling_stances):
        expected_speed = np.cos(2 * alpha) * np.sqrt(
            stance.omega[0] ** 2
            + 4 * params["gravity"] / params["length"] * np.sin(gamma) * np.sin(alpha)
        )
        assert stance.after_state[1] == pytest.approx(expected_speed, abs=2e-9)
        accumulated_loss += stance.loss
        assert stance.after_energy == pytest.approx(
            rolling_stances[0].energy[0] - accumulated_loss, abs=1e-8
        )
        if index + 1 < len(rolling_stances):
            next_stance = rolling_stances[index + 1]
            assert next_stance.time[0] == stance.time[-1]
            np.testing.assert_allclose(
                [next_stance.theta[0], next_stance.omega[0]], stance.after_state
            )
            np.testing.assert_array_equal(next_stance.foot, stance.after_foot)
            assert next_stance.energy[0] == pytest.approx(stance.after_energy)


def test_nondefault_physical_parameters():
    params = generate_params()
    params.update(mass=2.3, length=0.7, gravity=3.71, num_spokes=10, slope_angle=0.1)
    stance = simulate(params, steps=1, initial_speed=1.2, samples_per_step=13)[0]
    np.testing.assert_allclose(stance.kinetic, 0.5 * 2.3 * 0.7**2 * stance.omega**2)
    np.testing.assert_allclose(
        stance.potential, 2.3 * 3.71 * 0.7 * np.cos(stance.theta)
    )
    assert stance.time.size == 13
    assert stance.after_potential == pytest.approx(stance.potential[-1])


@pytest.mark.parametrize(
    "key,value",
    [
        ("mass", 0),
        ("length", -1),
        ("gravity", np.inf),
        ("num_spokes", 4),
        ("num_spokes", 8.5),
        ("slope_angle", 0),
        ("slope_angle", 0.5),
    ],
)
def test_invalid_physical_parameters(key, value):
    params = generate_params()
    params[key] = value
    with pytest.raises(ValueError, match=key):
        simulate(params)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0},
        {"samples_per_step": 1},
        {"initial_speed": 0.1},
        {"initial_speed": np.nan},
    ],
)
def test_invalid_simulation_arguments(kwargs):
    with pytest.raises(ValueError):
        simulate(**kwargs)


def test_eventual_stall_is_reported():
    params = generate_params()
    params["slope_angle"] = 0.001
    with pytest.raises(RuntimeError, match="lacks enough energy"):
        simulate(params, steps=10, initial_speed=1.5)
