"""Event-resolved downhill motion and energy accounting for a rimless wheel.

Angles are measured from the world vertical.  The stance-dependent quantity
``H = K + m g l cos(theta)`` defines the phase-plane contours, while the physical
potential energy also includes the height of the current stance foot.  Keeping
that height is essential: the hub does not jump when a new foot contacts the
slope, so physical potential energy is continuous at impact.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .rimless_wheel import contact_angles, dynamics, generate_params, reset_state


@dataclass(frozen=True)
class Stance:
    """One continuous stance, including both contact endpoints and its reset."""

    time: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    foot: np.ndarray
    kinetic: np.ndarray
    potential: np.ndarray
    energy: np.ndarray
    local_energy: np.ndarray
    after_state: np.ndarray
    after_foot: np.ndarray
    after_kinetic: float
    after_potential: float
    after_energy: float
    loss: float


def simulate(params=None, steps=5, initial_speed=1.5, samples_per_step=64) -> list[Stance]:
    """Return complete downhill stances with accurately located impact events.

    ``initial_speed`` is the positive angular velocity immediately after a
    contact reset.  Each stance contains ``samples_per_step`` equally spaced
    physical times, including its exact start and impact time.  Consecutive
    stances therefore share an endpoint time, but their states differ by the
    instantaneous contact reset.  Energies use SI units and the fixed datum
    ``y = 0`` at the first stance foot.

    This rolling demonstration requires ``0 < slope_angle < pi / num_spokes``
    and enough kinetic energy to pass the vertical configuration on every
    requested step.  An error is raised if the wheel would stall or fail to
    reach its next contact.
    """
    params = generate_params() if params is None else dict(params)
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
    for name, value, minimum in (
        ("steps", steps, 1),
        ("samples_per_step", samples_per_step, 2),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer of at least {minimum}.")

    mass, length, gravity = (params[key] for key in ("mass", "length", "gravity"))
    alpha = np.pi / spokes
    gamma = params["slope_angle"]
    if not np.isfinite(gamma) or not 0 < gamma < alpha:
        raise ValueError("slope_angle must satisfy 0 < slope_angle < pi / num_spokes.")
    lower, upper = contact_angles(params)
    minimum_speed = np.sqrt(2 * gravity / length * (1 - np.cos(lower)))
    if not np.isfinite(initial_speed) or initial_speed <= minimum_speed:
        raise ValueError(
            "initial_speed must exceed "
            f"{minimum_speed:.6g} rad/s to pass the vertical position."
        )

    def contact(t, state):
        return state[0] - upper

    contact.terminal = True
    contact.direction = 1

    foot_shift = 2 * length * np.sin(alpha) * np.array([np.cos(gamma), -np.sin(gamma)])
    time_scale = np.sqrt(length / gravity)
    foot = np.zeros(2)
    state = np.array([lower, initial_speed], dtype=float)
    start_time = 0.0
    stances = []
    for step in range(steps):
        if 0.5 * length / gravity * state[1] ** 2 + np.cos(state[0]) <= 1:
            raise RuntimeError(
                f"The wheel lacks enough energy to pass vertical on stance {step + 1}. "
                "Increase initial_speed or slope_angle, or request fewer steps."
            )
        solution = solve_ivp(
            lambda t, y: dynamics(t, y, params),
            (start_time, start_time + 100 * time_scale),
            state,
            events=contact,
            dense_output=True,
            rtol=1e-10,
            atol=1e-12,
            max_step=time_scale / 8,
        )
        if not solution.success or not solution.t_events[0].size:
            raise RuntimeError(
                f"No downhill impact was reached on stance {step + 1}: {solution.message}"
            )

        impact_time = float(solution.t_events[0][0])
        time = np.linspace(start_time, impact_time, samples_per_step)
        theta, omega = solution.sol(time)
        kinetic = 0.5 * mass * length**2 * omega**2
        local_potential = mass * gravity * length * np.cos(theta)
        potential = local_potential + mass * gravity * foot[1]
        energy = kinetic + potential
        local_energy = kinetic + local_potential

        after_state = reset_state(np.array([theta[-1], omega[-1]]), params, 1)
        after_foot = foot + foot_shift
        after_kinetic = float(0.5 * mass * length**2 * after_state[1] ** 2)
        after_potential = float(
            mass * gravity * (after_foot[1] + length * np.cos(after_state[0]))
        )
        after_energy = after_kinetic + after_potential
        loss = float(kinetic[-1] - after_kinetic)
        stances.append(
            Stance(
                time=time,
                theta=theta,
                omega=omega,
                foot=foot.copy(),
                kinetic=kinetic,
                potential=potential,
                energy=energy,
                local_energy=local_energy,
                after_state=after_state,
                after_foot=after_foot,
                after_kinetic=after_kinetic,
                after_potential=after_potential,
                after_energy=after_energy,
                loss=loss,
            )
        )
        state, foot, start_time = after_state, after_foot, impact_time
    return stances
