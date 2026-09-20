"""
Assignment 2: Inverted Pendulum Walker
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from models import inverted_pendulum_walker as model
from utilities import pd_controller, poincare, roa, simulation

# Model parameters and controller gains
params = {
    "gravity": 9.81,  # m/s^2
    "length": 1.0,  # m
    "mass": 1.0,  # kg
    "incline": 0.06,  # rad
    "angle_of_attack": np.pi / 8,  # rad
    "ankle_torque": 0.0,  # N m
    "balance_kp": 4,  # s^(-2)
    "balance_kd": 4,  # s^(-1)
}

initial_state = np.array([0.0, 3.0])
timestep = 1e-4
sim_time = 3.0

# test
# initial_state = np.array([0.02, 0.0])
# sim_time = 5.0

# RoA grid values -- 9 and 13 // 17 and 25
theta_values = np.linspace(-0.1, 0.1, 17)
theta_dot_values = np.linspace(-0.4, 0.4, 25)
roa_params = params.copy()
roa_params["angle_of_attack"] = np.pi / 8

angle_bounds = (
    params["incline"] - np.pi / 2,
    params["incline"] + roa_params["angle_of_attack"],
)


def choose_alpha(speed, return_table, step_policy):
    """
    Return angle of attack at the nearest sampled speed
    """
    speed_index = poincare._nearest_speed_index(speed, return_table["speed_values"])

    alpha = step_policy["best_alpha"][speed_index]
    if speed_index is None:
        return None

    if not np.pi / 8 <= alpha <= np.pi / 7:
        return None
    return float(alpha)


def find_control_event(model, trajectory, params, roa_data, event_time_tol):
    """
    Return the first RoA entry or midstance crossing, or None
    -- help decide whether to turn on ankle control for balance
    or to choose next alpha
    """

    times = trajectory["time_traj"]
    states = trajectory["state_traj"]
    impact_count = 0

    def roa_entry_guard(previous_state, next_state, params):
        return roa.event_guard(previous_state, next_state, roa_data)

    for index in range(1, len(times)):
        previous_time = times[index - 1]
        previous_state = states[:, index - 1]
        time = times[index]
        state = states[:, index]

        # Count instantaneous impact resets
        impact_reset = time == previous_time
        if impact_reset:
            impact_count += 1

        if not np.all(np.isfinite(state)):
            return None

        # reset is not a midstance crossing
        reached_midstance = (
            not impact_reset
            and poincare.event_guard(previous_state, state)
        )

        if reached_midstance:
            time, state = poincare.locate_event(
                model, poincare.event_guard,
                previous_time, previous_state,
                time - previous_time, params, event_time_tol,
            )

        # Check if RoA entry happened before or at this time
        entered_roa = roa.is_in_roa(state, roa_data)

        if entered_roa and not impact_reset:
            time, state = poincare.locate_event(
                model, roa_entry_guard,
                previous_time, previous_state,
                time - previous_time, params, event_time_tol,
            )
        elif reached_midstance:
            state = state.copy()
            state[0] = 0.0
            entered_roa = roa.is_in_roa(state, roa_data)

        if entered_roa:
            return "capture", time, state.copy(), impact_count

        if reached_midstance:
            return "midstance", time, state, impact_count

    return None


def simulate_walking(
    model, initial_state, params, roa_data, return_table, step_policy,
    timestep, sim_time, event_time_tol=1e-8,
):
    """
    Simulate walking and switch to balancing after entering the RoA
    """
    state = np.array(initial_state)
    controls = params.copy()
    time = 0.0
    step = 0
    end_time = 0.0
    outcome = "time_limit"

    balancing = roa.is_in_roa(state, roa_data)
    capture_time = None
    if balancing:
        capture_time = 0.0

    times, states = [], []
    alphas, torques = [], []
    impacts = []
    policy_decisions = []

    gamma = controls["incline"]
    angle_bounds = (gamma - np.pi / 2, gamma + np.pi / 2)

    while True:
        # Choose the controls for the current state
        if outcome == "time_limit":
            controls["ankle_torque"] = 0.0

            if balancing:
                controls["ankle_torque"] = pd_controller.compute_ankle_torque(
                    state, controls
                )

            elif state[0] == 0.0 and state[1] > 0.0:
                alpha = choose_alpha(state[1], return_table, step_policy)

                if alpha is None:
                    outcome = "failed"
                else:
                    controls["angle_of_attack"] = alpha
                    policy_decisions.append({
                        "time": time,
                        "state": state.copy(),
                        "alpha": alpha,
                    })

        # Record the initial state, each completed timestep, or a failure
        if time >= end_time or outcome != "time_limit":
            times.append(time)
            states.append(state.copy())
            alphas.append(controls["angle_of_attack"])
            torques.append(controls["ankle_torque"])

        if time >= sim_time or outcome != "time_limit":
            break

        # Move to the next timestep only after finishing the current one
        if time >= end_time:
            step += 1
            end_time = min(step * timestep, sim_time)

        trial = simulation.advance_timestep(
            model, time, state, end_time - time, controls,
            angle_bounds=angle_bounds,
            impact_time_tol=event_time_tol,
        )

        event = None
        if not balancing:
            event = find_control_event(
                model, trial, controls, roa_data, event_time_tol
            )

        if event is None:
            # Keep the whole trial
            time = trial["time_traj"][-1]
            state = trial["state_traj"][:, -1]
            impacts.extend(trial["impacts"])

            if trial["outcome"] != "completed":
                outcome = trial["outcome"]

        else:
            # Keep only the part of the trial up to the control event
            kind, time, state, impact_count = event
            impacts.extend(trial["impacts"][:impact_count])

            if kind == "capture":
                balancing = True
                capture_time = time

            # For midstance, find_control_event sets theta to zero
            # The next loop iteration therefore chooses a new alpha

    return {
        "time_traj": np.array(times),
        "state_traj": np.array(states).T,
        "alpha_traj": np.array(alphas),
        "torque_traj": np.array(torques),
        "impacts": impacts,
        "policy_decisions": policy_decisions,
        "capture_time": capture_time,
        "outcome": outcome,
    }


roa_data = roa.estimate_roa(
    model,
    pd_controller.compute_ankle_torque,
    roa_params,
    theta_values,
    theta_dot_values,
    timestep=timestep,
    sim_time=5.0,
    angle_bounds=angle_bounds
)

speed_values = np.linspace(0.0, np.sqrt(2 * params["gravity"] / params["length"]), 11)
alpha_values = np.linspace(np.pi / 8, np.pi / 7, 5)

return_table = poincare.build_return_table(
    model,
    params,
    roa_data,
    speed_values,
    alpha_values,
    timestep=0.001,
    sim_time=5.0,
)

step_policy = poincare.compute_step_policy(return_table)

walking_result = simulate_walking(
    model, initial_state, params, roa_data, return_table, step_policy,
    timestep=timestep, sim_time=sim_time,
)
time_traj = walking_result["time_traj"]
state_traj = walking_result["state_traj"]
alpha_traj = walking_result["alpha_traj"]
torque_traj = walking_result["torque_traj"]
completed_steps = len(walking_result["impacts"])
capture_time = walking_result["capture_time"]

# Potential energy is measured from a fixed height reference -- for plotting
stance_height = 0.0
for impact in walking_result["impacts"]:
    theta_minus = impact["state_minus"][0]
    theta_plus = impact["state_plus"][0]
    stance_height += params["length"] * (
        np.cos(theta_minus) - np.cos(theta_plus)
    )
energy = (
    model.calculate_energy(state_traj[:, -1], params)
    + params["mass"] * params["gravity"] * stance_height
)


fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")


def draw_frame(index):
    # The massless swing leg is repositioned instantaneously at each impact
    frame_params = params.copy()
    frame_params["angle_of_attack"] = alpha_traj[index]
    frame_params["ankle_torque"] = torque_traj[index]
    model.visualize(state_traj[:, index], frame_params, ax=ax)
    ax.set_title(f"t = {time_traj[index]:.2f} s")


# Simulate at a small timestep, but render only 25 frames per second.
fps = 25
frame_stride = round(1 / (fps * timestep))
frame_indices = list(range(0, time_traj.size, frame_stride))
if frame_indices[-1] != time_traj.size - 1:
    frame_indices.append(time_traj.size - 1)

animation = FuncAnimation(
    fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
)
output = project_root / "output" / "assignment_2"
output.mkdir(parents=True, exist_ok=True)
animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))

# To save an MP4 instead, install FFmpeg and use:
# animation.save(output / "walker.mp4", writer="ffmpeg", fps=fps)
print(f"Saved {output / 'walker.gif'} ({completed_steps} footstrikes).")

# Plot RoA
converged = roa_data["converged"]
colors = ["lightgray", "green"]

roa_fig, roa_ax = plt.subplots(
    figsize=(8, 5), layout="constrained"
)

roa_ax.pcolormesh(
    roa_data["theta_values"],
    roa_data["theta_dot_values"],
    converged.astype(int),
    cmap=ListedColormap(colors),
    vmin=0,
    vmax=1,
    shading="nearest",
)

roa_ax.axhline(0, color="black", linewidth=0.5)
roa_ax.axvline(0, color="black", linewidth=0.5)

roa_ax.set_xlabel(r"Initial angle $\theta_0$ (rad)")
roa_ax.set_ylabel(r"Initial angular velocity $\dot{\theta}_0$ (rad/s)")
roa_ax.set_title("Walker: sampled balancing region of attraction")

roa_ax.legend(
    handles=[
        Patch(color=colors[0], label="Convergence not confirmed"),
        Patch(color=colors[1], label="Converged to upright"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
)

roa_fig.savefig(output / "roa.png", dpi=300, bbox_inches="tight")

# Plot the outcome for each initial speed and alpha
outcome_styles = [
    ("captured", "Reached standing RoA", "green"),
    ("returned", "Returned to midstance", "royalblue"),
    ("failed", "Failed trial", "tomato"),
    ("unresolved", "Time limit reached", "lightgray"),
]

outcome_codes = np.full(return_table["outcomes"].shape, np.nan)

for code, (outcome, label, color) in enumerate(outcome_styles):
    outcome_codes[return_table["outcomes"] == outcome] = code

table_fig, table_ax = plt.subplots(
    figsize=(8, 5), layout="constrained"
)

table_ax.pcolormesh(
    return_table["alpha_values"],
    return_table["speed_values"],
    outcome_codes,
    cmap=ListedColormap(
        [color for outcome, label, color in outcome_styles]
    ),
    vmin=-0.5,
    vmax=len(outcome_styles) - 0.5,
    shading="nearest",
)

table_ax.set_xlabel(r"Angle of attack $\alpha$ (rad)")
table_ax.set_ylabel(
    r"Initial midstance angular velocity $\dot{\theta}_k$ (rad/s)"
)
table_ax.set_title("Walker: state–action table")

table_ax.set_xlim(
    return_table["alpha_values"][0],
    return_table["alpha_values"][-1],
)
table_ax.set_ylim(
    return_table["speed_values"][0],
    return_table["speed_values"][-1],
)

table_ax.legend(
    handles=[
        Patch(color=color, label=label)
        for outcome, label, color in outcome_styles
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
)

table_fig.savefig(
    output / "return_table.png",
    dpi=300,
    bbox_inches="tight",
)


# Plot the midstance Poincare return map
return_fig, return_ax = plt.subplots(
    figsize=(8, 6), layout="constrained"
)

speeds = return_table["speed_values"]

for column, alpha in enumerate(return_table["alpha_values"]):
    next_speeds = return_table["next_speeds"][:, column]

    return_ax.plot(
        speeds,
        next_speeds,
        marker="o",
        markersize=4,
        linewidth=1.5,
        label=rf"$\alpha = {alpha:.4f}$ rad",
    )

return_ax.plot(
    speeds,
    speeds,
    color="black",
    linestyle="--",
    linewidth=1,
    label=r"$\dot{\theta}_{k+1} = \dot{\theta}_k$",
)

return_ax.set_xlabel(
    r"Initial midstance speed $\dot{\theta}_k$ (rad/s)"
)
return_ax.set_ylabel(
    r"Next midstance speed $\dot{\theta}_{k+1}$ (rad/s)"
)
return_ax.set_title("Walker: midstance Poincaré return map")

return_ax.set_xlim(0, speeds[-1])
return_ax.set_ylim(bottom=0)
return_ax.set_aspect("equal", adjustable="box")
return_ax.grid(True, alpha=0.3)
return_ax.legend()

return_fig.savefig(
    output / "poincare_return_map.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
