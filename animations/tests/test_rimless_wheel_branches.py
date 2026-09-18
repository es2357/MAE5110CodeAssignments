"""Physical checks for all three single-impact rimless-wheel motions."""

import numpy as np
import pytest

from animation_models.rimless_wheel import contact_angles, generate_params
from animation_models.rimless_wheel_branches import simulate_case, simulate_cases


@pytest.fixture(scope="module")
def cases():
    return simulate_cases()


def test_first_impact_follows_each_of_the_three_energy_branches(cases):
    params = generate_params()
    lower, upper = contact_angles(params)
    alpha, gamma = np.pi / params["num_spokes"], params["slope_angle"]
    c = np.cos(2 * alpha)
    delta = 4 * params["gravity"] / params["length"] * np.sin(alpha) * np.sin(gamma)
    expected = [c * np.sqrt(1.5**2 + delta), -c * 0.9, -c * np.sqrt(2.3**2 - delta)]
    assert [case.key for case in cases] == ["A", "B", "C"]
    assert [case.direction for case in cases] == [1, -1, -1]
    for case, initial, final, after in zip(
        cases, [lower, lower, upper], [upper, lower, lower], expected, strict=True
    ):
        np.testing.assert_allclose(case.theta[[0, -1]], [initial, final], atol=1e-12)
        assert case.omega[0] == case.initial_speed
        assert case.direction * case.omega[-1] > 0
        assert case.after_state[1] == pytest.approx(after, abs=2e-9)
        # Interior samples stay strictly inside the stance: no earlier
        # collision was skipped in order to reach the reported endpoint.
        assert np.all(case.theta[1:-1] > lower)
        assert np.all(case.theta[1:-1] < upper)
        if case.key != "B":
            assert np.all(case.direction * np.diff(case.theta) > 0)


def test_stance_energy_is_conserved_and_impact_loses_only_kinetic_energy(cases):
    params = generate_params()
    mass, length, gravity = (params[name] for name in ("mass", "length", "gravity"))
    alpha = np.pi / params["num_spokes"]
    for case in cases:
        kinetic = 0.5 * mass * length**2 * case.omega**2
        potential = mass * gravity * (case.foot[1] + length * np.cos(case.theta))
        energy = kinetic + potential
        np.testing.assert_allclose(energy, energy[0], rtol=0, atol=2e-9)
        after_kinetic = 0.5 * mass * length**2 * case.after_state[1] ** 2
        after_potential = (
            mass * gravity * (case.after_foot[1] + length * np.cos(case.after_state[0]))
        )
        assert after_potential == pytest.approx(potential[-1], abs=1e-12)
        assert kinetic[-1] - after_kinetic == pytest.approx(
            kinetic[-1] * np.sin(2 * alpha) ** 2, abs=1e-12
        )
        assert after_kinetic + after_potential < energy[-1]


def test_reversal_and_upright_are_actual_sampled_events(cases):
    params = generate_params()
    a, b, c = cases
    assert a.turn_index is None
    assert c.turn_index is None
    assert b.upright_index is None
    assert b.turn_index is not None
    turn = b.turn_index
    assert 0 < turn < b.time.size - 1
    assert b.omega[turn] == 0
    assert np.all(b.omega[:turn] > 0)
    assert np.all(b.omega[turn + 1 :] < 0)
    assert np.all(b.theta < 0)
    expected_turn = -np.arccos(
        np.cos(b.theta[0])
        + params["length"] * b.initial_speed**2 / (2 * params["gravity"])
    )
    assert b.theta[turn] == pytest.approx(expected_turn, abs=2e-10)
    # The genuine rocking return keeps the negative impact velocity. A
    # settling shortcut would incorrectly replace this event with zero.
    assert b.omega[-1] == pytest.approx(-b.initial_speed, abs=2e-9)
    for case in (a, c):
        index = case.upright_index
        assert index is not None
        assert 0 < index < case.time.size - 1
        assert case.theta[index] == 0
        assert case.direction * case.theta[index - 1] < 0
        assert case.direction * case.theta[index + 1] > 0
        expected_squared = case.initial_speed**2 - 2 * params["gravity"] / params[
            "length"
        ] * (1 - np.cos(case.theta[0]))
        assert case.omega[index] ** 2 == pytest.approx(expected_squared, abs=2e-9)


def test_time_and_contact_geometry_are_continuous(cases):
    params = generate_params()
    length, gamma = params["length"], params["slope_angle"]
    ground_normal = np.array([np.sin(gamma), np.cos(gamma)])
    for case in cases:
        assert case.time[0] == 0
        assert case.time[-1] > 0
        assert np.all(np.diff(case.time) > 0)
        assert case.time.size == case.theta.size == case.omega.size
        assert case.time.size >= 81
        hub_before = case.foot + length * np.array(
            [np.sin(case.theta[-1]), np.cos(case.theta[-1])]
        )
        hub_after = case.after_foot + length * np.array(
            [np.sin(case.after_state[0]), np.cos(case.after_state[0])]
        )
        np.testing.assert_allclose(hub_before, hub_after, atol=1e-12)
        assert case.foot @ ground_normal == pytest.approx(0, abs=1e-12)
        assert case.after_foot @ ground_normal == pytest.approx(0, abs=1e-12)
        assert case.direction * (case.after_foot[0] - case.foot[0]) > 0


def test_sampling_resolution_does_not_change_first_impact():
    params = generate_params()
    sparse = simulate_case(0.9, params, samples_per_case=3)
    dense = simulate_case(0.9, params, samples_per_case=173)
    assert sparse.time[-1] == dense.time[-1]
    np.testing.assert_array_equal(sparse.after_state, dense.after_state)
    assert sparse.time[sparse.turn_index] == dense.time[dense.turn_index]


@pytest.mark.parametrize("direction", [1, -1])
def test_upright_threshold_has_no_finite_impact(direction):
    params = generate_params()
    angle = contact_angles(params)[0 if direction == 1 else 1]
    speed = direction * np.sqrt(
        2 * params["gravity"] / params["length"] * (1 - np.cos(angle))
    )
    with pytest.raises(ValueError, match="no finite-time return"):
        simulate_case(speed, params)


@pytest.mark.parametrize(
    "changes", [{"slope_angle": 0.2}, {"length": 0.3}, {"length": 0.1}]
)
def test_case_labels_cannot_silently_change_branch(changes):
    params = {**generate_params(), **changes}
    with pytest.raises(ValueError, match="These cases require"):
        simulate_cases(params)
