import sys
import timeit
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Direct execution needs the project root to find the shared packages.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integrators import explicit_euler as integrator
from models import pendulum as model

# from integrators import rk4 as integrator

# Basic simulation of the pendulum

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # rod length (m)
    "mass": 0.2,  # point mass at end of rod (kg)
    "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
}


# set-up
initial_state = np.array([np.pi / 4, 0.0])

timestep = 1e-5
sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state


start_time = timeit.default_timer()

for step, t in enumerate(time_traj[:-1]):
    state_traj[:, step + 1] = integrator.step(
        model.dynamics,
        t,
        state_traj[:, step],
        timestep,
        params,
    )

elapsed_time = timeit.default_timer() - start_time

print(f"Integrator runtime: {elapsed_time:.6f} seconds")

# simulation loop -- explicit euler method

# for step, t in enumerate(time_traj[:-1]):
#     state_traj[:, step + 1] = state_traj[:, step] + timestep * model.dynamics(
#         t, state_traj[:, step], params
#     )

# simulation loop -- with integrator
for step, t in enumerate(time_traj[:-1]):
    state_traj[:, step + 1] = integrator.step(
        model.dynamics,
        t,
        state_traj[:, step],
        timestep,
        params,
    )


# sanity check the energies: since there is no actuation, and no damping, total energy should stay
# constant. If we turn on the damping coefficient, it should slowly bleed out energy until it comes to
# a stand-still.

kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)

plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Pendulum energy")
plt.legend()
plt.tight_layout()
plt.show()

# TODO: make a phase portrait plot
