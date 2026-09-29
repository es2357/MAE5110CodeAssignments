import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from integrators import rk4 as integrator
from models import inverted_pendulum_walker as model
import stabilizing_upright as controller

# Fixed controls for this visualization example.
params = {
    "gravity": 9.81,  # m/s^2
    "length": 1.0,  # m
    "mass": 1.0,  # kg
    "incline": 0.06,  # rad
    "angle_of_attack": np.pi / 8,  # rad
    "ankle_torque": 0.0,  # N m
}


timestep = 1e-3
sim_time = 20.0

n_timesteps = round(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep

theta_grid = np.linspace(-0.4, 0.6, 100)
angular_vel_grid = np.linspace(-1.6, 1.6, 100)

tol = 1e-3
settling_time = 0.5
settling_steps = round(settling_time / timestep)

theta_cur, angular_vel_cur = np.meshgrid(theta_grid, angular_vel_grid)

settle_count = np.zeros(theta_cur.shape, dtype=int)
converged = np.zeros(theta_cur.shape, dtype=bool)

gamma = params["incline"]
alpha = params["angle_of_attack"]

backward_guard = gamma - alpha
forward_guard = gamma + alpha

state = np.stack([theta_cur, angular_vel_cur])
failed = np.zeros(theta_cur.shape, dtype=bool)

initial_guards = (
    (theta_cur <= backward_guard)
    |
    (theta_cur >= forward_guard)
)
failed |= initial_guards

for step, t in enumerate(time_traj[:-1]):
    params["ankle_torque"] = controller.feedback_linearization(state, params)
    next_state = integrator.integrate(
        t,
        state,
        timestep,
        model.dynamics,
        params
    )
    
    crossed_guard = (
        (next_state[0] <= backward_guard)
        |
        (next_state[0] >= forward_guard)
    )
    
    failed_new = (~converged) & crossed_guard
    failed |= failed_new
    
    inside_upright = controller.is_upright(
        next_state,
        tol
    )
    
    valid_upright = ((~failed) &
                     (~converged) &
                    inside_upright &
                    (next_state[0] > backward_guard) &
                    (next_state[0] < forward_guard)
    )
    
    settle_count = np.where(
        valid_upright,
        settle_count + 1,
        0
    )
    
    converged_new = ((~failed) & (~converged) & (settle_count >= settling_steps))
    converged |= converged_new
    
    state = next_state

roa = converged & (~failed)

for j, th in enumerate(theta_grid):
    converged_omegas = angular_vel_grid[roa[:, j]]
    if converged_omegas.size:
        print(f"theta={th:.3f}: omega in [{converged_omegas.min():.3f}, {converged_omegas.max():.3f}]")

plt.figure(figsize=(8,6))

plt.contourf(
    theta_cur,
    angular_vel_cur,
    roa.astype(int),
    levels = [-0.5, 0.5, 1.5],
    colors=["blue","green"]
)

plt.xlabel(r"$\theta$ [rad]")
plt.ylabel(r"$\dot{\theta}$ [rad/s]")
plt.title("Region of Attraction (RoA)")

legend_elements = {
    Patch(facecolor="green", label = "converged"),
    Patch(facecolor="blue", label = "not converged / timed out")
}
plt.legend(handles=legend_elements)

plt.show()

np.savez(
    "roa_data.npz",
    theta_grid=theta_grid,
    angular_vel_grid=angular_vel_grid,
    roa=roa
)


