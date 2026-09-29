import numpy as np
import matplotlib.pyplot as plt

from models import bouncingBall as model
from integrators import explicit_euler as integrator

# Parameters

params = {
    "gravity": 9.81,
    "mass": 0.2,
    "restitution": 1,
}

initial_state = np.array([
    5.0,    # initial position [m]
    0.0     # initial velocity [m/s]
])

sim_time = 5.0
timestep = 0.001



# Time array

time_traj = np.arange(
    0,
    sim_time + timestep,
    timestep
)

n_timesteps = len(time_traj)


# State trajectory

state_traj = np.zeros((2, n_timesteps))

state_traj[:, 0] = initial_state


# Integration

for step, t in enumerate(time_traj[:-1]):

    # Integrate one timestep
    state_traj[:, step + 1] = integrator.integrate(
        t,
        state_traj[:, step],
        timestep,
        model.dynamics,
        params
    )

    # Check for ground contact
    if state_traj[0, step + 1] <= 0:

        # Collision correction
        state_traj[0, step + 1] = 0

        state_traj[1, step + 1] = (
            -params["restitution"]
            * state_traj[1, step + 1]
        )


# Calculate energy

kinetic_energy = np.zeros(n_timesteps)
potential_energy = np.zeros(n_timesteps)
total_energy = np.zeros(n_timesteps)

for step in range(n_timesteps):

    kinetic_energy[step], potential_energy[step] = (
        model.calculate_energy(
            state_traj[:, step],
            params
        )
    )

    total_energy[step] = (
        kinetic_energy[step]
        + potential_energy[step]
    )


# Plot position

plt.figure()

plt.plot(
    time_traj,
    state_traj[0, :]
)

plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.title("Bouncing Ball Position")
plt.grid()

plt.show()


# Plot velocity

plt.figure()

plt.plot(
    time_traj,
    state_traj[1, :]
)

plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("Bouncing Ball Velocity")
plt.grid()

plt.show()


# Plot energy

plt.figure()

plt.plot(
    time_traj,
    kinetic_energy,
    label="Kinetic Energy"
)

plt.plot(
    time_traj,
    potential_energy,
    label="Potential Energy"
)

plt.plot(
    time_traj,
    total_energy,
    label="Total Energy"
)

plt.xlabel("Time [s]")
plt.ylabel("Energy [J]")
plt.title("Bouncing Ball Energy")
plt.legend()
plt.grid()

plt.show()


# Phase portrait

plt.figure()

plt.plot(
    state_traj[0, :],
    state_traj[1, :]
)

plt.xlabel("Position [m]")
plt.ylabel("Velocity [m/s]")
plt.title("Bouncing Ball Phase Portrait")
plt.grid()

plt.show()