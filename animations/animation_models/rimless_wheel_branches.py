"""Single-impact motions illustrating the three signed return-map branches."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .rimless_wheel import contact_angles, dynamics, generate_params, reset_state


@dataclass(frozen=True)
class BranchMotion:
    """A continuous stance and the reset at its first physical collision.

    ``time`` contains uniform samples plus any exact turning/upright event
    times. ``foot`` is the fixed world position of the stance foot; the
    instantaneous reset is stored separately from the pre-impact samples.
    """

    key: str
    initial_speed: float
    direction: int
    time: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    foot: np.ndarray
    after_state: np.ndarray
    after_foot: np.ndarray
    turn_index: int | None
    upright_index: int | None


def _validated_params(params):
    """Check the physical regime used by these three stance illustrations."""
    params = dict(params)
    for name in ("mass", "length", "gravity"):
        if not np.isfinite(params[name]) or params[name] <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    spokes = params["num_spokes"]
    if (
        isinstance(spokes, (bool, np.bool_))
        or not np.isfinite(spokes)
        or spokes != int(spokes)
        or spokes < 5
    ):
        raise ValueError("num_spokes must be an integer of at least 5.")
    alpha = np.pi / spokes
    gamma = params["slope_angle"]
    if not np.isfinite(gamma) or not 0 <= gamma < alpha:
        raise ValueError("slope_angle must satisfy 0 <= slope_angle < pi / num_spokes.")
    return params


def simulate_case(initial_speed, params, samples_per_case=81, key="") -> BranchMotion:
    """Integrate one signed stance through its first impact, including reversal.

    Positive velocities start at gamma-alpha; negative velocities start at
    gamma+alpha. The inward motion at the initial contact is not a new impact.
    ``samples_per_case`` specifies uniform samples, including both endpoints;
    exact turning and upright times are added when they occur between them.
    """
    params = _validated_params(params)
    alpha, gamma = np.pi / params["num_spokes"], params["slope_angle"]
    if (
        isinstance(samples_per_case, (bool, np.bool_))
        or not isinstance(samples_per_case, (int, np.integer))
        or samples_per_case < 2
    ):
        raise ValueError("samples_per_case must be an integer of at least 2.")
    if not np.isfinite(initial_speed) or initial_speed == 0:
        raise ValueError("initial_speed must be finite and nonzero.")

    lower, upper = contact_angles(params)
    start_angle = lower if initial_speed > 0 else upper
    barrier_speed = np.sqrt(
        2 * params["gravity"] / params["length"] * (1 - np.cos(start_angle))
    )
    if np.isclose(abs(initial_speed), barrier_speed, rtol=0, atol=1e-12):
        raise ValueError("The upright threshold has no finite-time return impact.")

    def lower_contact(t, state):
        return state[0] - lower

    def upper_contact(t, state):
        return state[0] - upper

    # Direction matters: the root at the starting boundary is an inward
    # departure, whereas a later outward crossing is a physical collision.
    lower_contact.terminal = upper_contact.terminal = True
    lower_contact.direction = -1
    upper_contact.direction = 1

    def turning(t, state):
        return state[1]

    def upright(t, state):
        return state[0]

    solution = solve_ivp(
        lambda t, state: dynamics(t, state, params),
        (0.0, 100 * np.sqrt(params["length"] / params["gravity"])),
        [start_angle, initial_speed],
        events=(lower_contact, upper_contact, turning, upright),
        dense_output=True,
        rtol=1e-10,
        atol=1e-12,
        max_step=0.02,
    )
    if not solution.success or not any(solution.t_events[i].size for i in (0, 1)):
        raise RuntimeError(f"No first contact was reached: {solution.message}")

    direction = -1 if solution.t_events[0].size else 1
    impact_time = float(solution.t_events[0 if direction == -1 else 1][0])
    special_times = [events[0] for events in solution.t_events[2:] if events.size]
    time = np.unique(
        np.concatenate(
            (
                np.linspace(0.0, impact_time, samples_per_case),
                special_times,
            )
        )
    )
    theta, omega = solution.sol(time)
    theta[0], omega[0] = start_angle, initial_speed
    theta[-1] = lower if direction == -1 else upper
    indices = [
        int(np.searchsorted(time, events[0])) if events.size else None
        for events in solution.t_events[2:]
    ]
    turn_index, upright_index = indices
    if turn_index is not None:
        omega[turn_index] = 0.0
    if upright_index is not None:
        theta[upright_index] = 0.0

    foot = np.zeros(2)
    after_state = reset_state([theta[-1], omega[-1]], params, direction)
    after_foot = foot + direction * 2 * params["length"] * np.sin(alpha) * np.array(
        [np.cos(gamma), -np.sin(gamma)]
    )
    return BranchMotion(
        key=key,
        initial_speed=float(initial_speed),
        direction=direction,
        time=time,
        theta=theta,
        omega=omega,
        foot=foot,
        after_state=after_state,
        after_foot=after_foot,
        turn_index=turn_index,
        upright_index=upright_index,
    )


def simulate_cases(params=None, samples_per_case=81) -> list[BranchMotion]:
    """Return A: downhill step, B: rocking reversal, C: uphill step.

    The chosen speeds illustrate these branches for the default Earth-gravity,
    eight-spoke wheel with gamma=0.08. Reject parameter changes that would
    change the branch associated with one of these labels.
    """
    params = _validated_params(generate_params() if params is None else params)
    lower, upper = contact_angles(params)
    scale = 2 * params["gravity"] / params["length"]
    omega_1 = np.sqrt(scale * (1 - np.cos(lower)))
    omega_2 = -np.sqrt(scale * (1 - np.cos(upper)))
    if not (0.9 < omega_1 < 1.5 and -2.3 < omega_2):
        raise ValueError(
            "These cases require A=1.5 > omega_1, 0 < B=0.9 < omega_1, "
            "and C=-2.3 < omega_2. With the supplied parameters, "
            f"omega_1={omega_1:.6g} and omega_2={omega_2:.6g} rad/s."
        )
    return [
        simulate_case(speed, params, samples_per_case, key)
        for key, speed in (("A", 1.5), ("B", 0.9), ("C", -2.3))
    ]
