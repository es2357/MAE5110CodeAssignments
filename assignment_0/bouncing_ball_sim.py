import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Direct execution needs the project root to find the shared packages.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integrators import rk4 as integrator
from models import bouncing_ball as model

params = {
    "gravity": 9.81,
    "mass": 0.2,
    "restitution": 0.8,
}

initial_state = np.array([1.0, 0.0])

timestep = 1e-5
sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep

state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

for step, t in enumerate(time_traj[:-1]):
    next_state = integrator.step(
        model.dynamics,
        t,
        state_traj[:, step],
        timestep,
        params,
    )

    state_traj[:, step + 1] = model.resolve_impact(
        next_state,
        params,
    )

print("Minimum height:", np.min(state_traj[0]))

kinetic_energy, potential_energy = model.calculate_energy(
    state_traj,
    params,
)
total_energy = kinetic_energy + potential_energy

fig, axes = plt.subplots(3, 1, sharex=True)

axes[0].plot(time_traj, state_traj[0])
axes[0].set_ylabel("Height (m)")

axes[1].plot(time_traj, state_traj[1])
axes[1].set_ylabel("Velocity (m/s)")

axes[2].plot(time_traj, total_energy)
axes[2].set_ylabel("Energy (J)")
axes[2].set_xlabel("Time (s)")

plt.tight_layout()
plt.show()
