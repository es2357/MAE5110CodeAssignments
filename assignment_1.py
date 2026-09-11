
# SIMULATION OF RIMLESS WHEEL
# (and plots + sanity checks)


import numpy as np
import matplotlib.pyplot as plt
import timeit

from models import rimless_wheel as model
from integrators import rk4 as integrator

params = model.generate_params()
alpha = np.pi / params["num_spokes"]
lower_angle, upper_angle = model.contact_angles(params)


# ------------------------------------------------------------
#               SANITY CHECK: test downhill impact
# ------------------------------------------------------------

state_minus = np.array([upper_angle, 2.0])

direction = model.impact_direction(state_minus, params)
state_plus = model.reset_state(state_minus, params, direction)

# Before impact: [0.47269908 2.        ]
# After impact:  [-0.31269908  1.41421356]
print("----------------------------------")
print("Sanity Check: one downhill impact")
print("Before impact:", state_minus)
print("After impact: ", state_plus)
print("----------------------------------")


np.testing.assert_allclose(
    state_plus,
    np.array([lower_angle, 2.0 * np.cos(2 * alpha)]),
)

assert model.impact_direction(state_plus, params) == 0
# ------------------------------------------------------------


# Finding when impact happens during a timestep -- RK4 misalignment
def find_impact_time(t, state, timestep, params, direction):
    """Find the time of impact during a timestep using bisection"""

    # if theta >= upper_angle and theta_dot > 0 -> impact downhill has happened
    lower_angle, upper_angle = model.contact_angles(params)

    if direction == 1:
        impact_angle = upper_angle
    else:
        impact_angle = lower_angle

    # Impact occurs somewhere between these two times
    t_minus = 0.0
    t_plus = timestep

    # Keep narrowing the interval until it is very small
    while t_plus - t_minus > 1e-8:

        t_mid = 0.5 * (t_minus + t_plus)

        state_mid = integrator.step(model.dynamics, t, state, t_mid, params)

        theta_mid = state_mid[0]

        if direction == 1:
            crossed = theta_mid >= impact_angle
        else:
            crossed = theta_mid <= impact_angle

        if crossed:
            t_plus = t_mid
        else:
            t_minus = t_mid

    return t_plus, impact_angle

# ------------------------------------------------------------
#                           one timestep
# ------------------------------------------------------------


MAX_IMPACTS_PER_STEP = 100


def should_settle(state, params):
    """Check whether the wheel has too little energy to roll over upright."""

    theta, theta_dot = state

    gravity = params["gravity"]
    length = params["length"]

    # Speed needed to reach theta = 0 from the current contact angle
    required_speed_squared = (
        2 * gravity / length
        * (1 - np.cos(theta))
    )

    return theta_dot**2 < required_speed_squared


def step(t, state, timestep, params):
    """Take one timestep, handling all impacts that occur during the step."""

    current_state = state.copy()
    current_time = t
    remaining_time = timestep

    impacts = []

    if len(impacts) > MAX_IMPACTS_PER_STEP:
        raise RuntimeError(
            "Too many impacts in one timestep."
        )

    while remaining_time > 0:

        # try integrating through all of the remaining time
        trial_state = integrator.step(
            model.dynamics,
            current_time,
            current_state,
            remaining_time,
            params,
        )

        direction = model.impact_direction(trial_state, params)

        # if no impact in the remaining time
        if direction == 0:
            return trial_state, impacts, False

        # find impact time and angle during the remaining time
        impact_dt, impact_angle = find_impact_time(
            current_time,
            current_state,
            remaining_time,
            params,
            direction,
        )

        # state immediately before impact
        state_minus = integrator.step(
            model.dynamics,
            current_time,
            current_state,
            impact_dt,
            params,
        )

        state_minus[0] = impact_angle

        # collision reset
        state_plus = model.reset_state(
            state_minus,
            params,
            direction,
        )

        impacts.append({
            "time": current_time + impact_dt,
            "direction": direction,
            "state_minus": state_minus.copy(),
            "state_plus": state_plus.copy(),
        })

        # Check whether the wheel has enough energy to
        # continue rolling over upright
        if should_settle(state_plus, params):

            state_plus[1] = 0.0

            return state_plus, impacts, True

        # Otherwise continue with the remaining timestep
        current_state = state_plus

        current_time += impact_dt
        remaining_time -= impact_dt

    return current_state, impacts, False


# ------------------------------------------------------------
# SANITY CHECK: continuous dynamics
# ------------------------------------------------------------

print("----------------------------------")
print("Sanity Check: continuous dynamics")

# Check acceleration at the beginning, middle, and end of a stance
test_angles = [lower_angle, 0.0, upper_angle]

for theta in test_angles:

    test_state = np.array([theta, 0.0])

    state_dot = model.dynamics(
        0.0,
        test_state,
        params,
    )

    theta_ddot = state_dot[1]

    print(
        f"theta = {theta:.3f} rad, "
        f"theta_ddot = {theta_ddot:.3f} rad/s^2"
    )

    # Check that theta_ddot has the expected value
    expected_theta_ddot = (
        params["gravity"]
        / params["length"]
        * np.sin(theta)
    )

    np.testing.assert_allclose(
        theta_ddot,
        expected_theta_ddot,
    )

print("Continuous dynamics passed.")
print("----------------------------------")


# ------------------------------------------------------------
#                           sim
# ------------------------------------------------------------

# Start immediately after a downhill impact
initial_state = np.array([lower_angle, 1.5])

timestep = 1e-3
sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep

state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

impact_log = []

start_time = timeit.default_timer()

for k, t in enumerate(time_traj[:-1]):

    next_state, impacts, settled = step(
        t,
        state_traj[:, k],
        timestep,
        params,
    )

    state_traj[:, k + 1] = next_state

    # Add every impact that occurred during this timestep
    impact_log.extend(impacts)

    if settled:
        # Hold the wheel at rest for the rest of the simulation
        state_traj[:, k + 1:] = next_state[:, None]
        print(f"Wheel settled at approximately t = {t:.3f} s")
        break

elapsed_time = timeit.default_timer() - start_time

print(f"Simulation runtime: {elapsed_time:.6f} seconds")
print(f"Number of impacts: {len(impact_log)}")


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------

# angle

plt.figure()

plt.plot(
    time_traj,
    state_traj[0],
    label=r"$\theta$",
)

plt.axhline(
    upper_angle,
    linestyle="--",
    color="red",
    label=r"$\gamma + \alpha$",
)

plt.axhline(
    lower_angle,
    linestyle="--",
    color="green",
    label=r"$\gamma - \alpha$",
)

for impact in impact_log:
    plt.axvline(
        impact["time"],
        linestyle=":",
        color="grey",
        alpha=0.7,
    )

plt.xlabel("Time (s)")
plt.ylabel(r"$\theta$ (rad)")
plt.title("Rimless wheel angle")
plt.legend()
plt.tight_layout()
plt.show()


# angular velocity
plt.figure()

plt.plot(
    time_traj,
    state_traj[1],
    label=r"$\dot{\theta}$",
)

for impact in impact_log:
    plt.axvline(
        impact["time"],
        linestyle=":",
        color="grey",
        alpha=0.7,
    )

plt.xlabel("Time (s)")
plt.ylabel(r"$\dot{\theta}$ (rad/s)")
plt.title("Rimless wheel angular velocity")
plt.legend()
plt.tight_layout()
plt.show()


# phase portrait

plt.figure()

plt.plot(
    state_traj[0],
    state_traj[1],
)

plt.xlabel(r"$\theta$ (rad)")
plt.ylabel(r"$\dot{\theta}$ (rad/s)")
plt.title("Rimless wheel phase portrait")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# SANITY CHECK: energy plots
# ------------------------------------------------------------

mass = params["mass"]
gravity = params["gravity"]
length = params["length"]
gamma = params["slope_angle"]

# Keep track of the height of the current contact point
# Set the first contact point to y = 0
contact_height = np.zeros(n_timesteps)

impact_index = 0
current_contact_height = 0.0

for k, t in enumerate(time_traj):

    # Check whether an impact occurred before this sample time
    while (
        impact_index < len(impact_log)
        and impact_log[impact_index]["time"] <= t
    ):

        direction = impact_log[impact_index]["direction"]

        # moving to the next downhill foot lowers the contact point
        current_contact_height -= (direction * 2 * length * np.sin(alpha) * np.sin(gamma))

        impact_index += 1

    contact_height[k] = current_contact_height


theta = state_traj[0]
theta_dot = state_traj[1]

# KE
kinetic_energy = (0.5 * mass * (length * theta_dot) ** 2)

# Global height of the hub
hub_height = (contact_height + length * np.cos(theta))

# PE
potential_energy = (mass * gravity * hub_height)

# Total energy
total_energy = (kinetic_energy + potential_energy)

plt.figure()

plt.plot(
    time_traj,
    kinetic_energy,
    label="Kinetic energy",
)

plt.plot(
    time_traj,
    potential_energy,
    label="Potential energy",
)

plt.plot(
    time_traj,
    total_energy,
    label="Total energy",
)

for impact in impact_log:
    plt.axvline(
        impact["time"],
        linestyle=":",
        color="grey",
        alpha=0.7,
    )

plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Rimless wheel energy")
plt.legend()
plt.tight_layout()
plt.show()
