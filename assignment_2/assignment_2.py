"""Assignment 2: Inverted Pendulum Walker"""

import argparse
import csv
import json
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
from utilities import grid_resolution, pd_controller, poincare, roa, simulation

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


def choose_alpha(speed, return_table, step_policy):
    """Return the policy angle at the nearest sampled speed, or None if invalid"""
    speed_index = poincare._nearest_speed_index(speed, return_table["speed_values"])

    if speed_index is None:
        return None

    alpha = step_policy["best_alpha"][speed_index]
    if not np.pi / 8 <= alpha <= np.pi / 7:
        return None
    return float(alpha)


def find_control_event(model, trajectory, params, roa_data, event_time_tol):
    """Return (kind, time, state, impact_count) for the first control event, or None"""
    times = trajectory["time_traj"]
    states = trajectory["state_traj"]
    impact_count = 0

    def roa_entry_guard(previous_state, next_state, params):
        """Return whether the states cross into the RoA"""
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

        # An impact reset is not a midstance crossing.
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
    """Return a dict of trajectories, impacts, decisions,
    control events, capture time, and outcome"""
    state = np.array(initial_state)
    controls = params.copy()
    time = 0.0
    step = 0
    end_time = 0.0
    outcome = "time_limit"

    balancing = roa.is_in_roa(state, roa_data)
    capture_time = 0.0 if balancing else None

    times, states = [], []
    alphas, torques = [], []
    impacts = []
    policy_decisions = []
    control_events = []
    if balancing:
        control_events.append({
            "kind": "capture", "time": 0.0,
            "state": state.copy(), "impact_count": 0,
        })

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
            control_events.append({
                "kind": kind, "time": time,
                "state": state.copy(), "impact_count": len(impacts),
            })

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
        "control_events": control_events,
        "capture_time": capture_time,
        "outcome": outcome,
    }


def build_control_table(grid, timestep):
    """Return the RoA estimate and speed–alpha return table"""
    n_theta, n_theta_dot, n_speed, n_alpha = grid
    balance_params = params.copy()
    balance_params["angle_of_attack"] = np.pi / 8
    print(f"Estimating RoA on {n_theta} x {n_theta_dot} samples...", flush=True)
    roa_data = roa.estimate_roa(
        model, pd_controller.compute_ankle_torque, balance_params,
        np.linspace(-0.1, 0.1, n_theta),
        np.linspace(-0.4, 0.4, n_theta_dot),
        timestep=timestep, sim_time=5.0,
        angle_bounds=(params["incline"] - np.pi / 2, params["incline"] + np.pi / 8),
    )
    print(f"Building {n_speed} x {n_alpha} return table...", flush=True)
    maximum_speed = np.sqrt(2 * params["gravity"] / params["length"])
    return_table = poincare.build_return_table(
        model, params, roa_data,
        np.linspace(0.0, maximum_speed, n_speed),
        np.linspace(np.pi / 8, np.pi / 7, n_alpha),
        timestep=timestep, sim_time=5.0,
    )
    return roa_data, return_table


def steps_before_capture(trial):
    """Return the footstrike count before RoA entry, or None if uncaptured"""
    capture_time = trial["capture_time"]
    if capture_time is None:
        return None
    return sum(impact["time"] <= capture_time for impact in trial["impacts"])


def save_walking_animation(trial, timestep, output):
    """Return the saved walker animation."""
    times = trial["time_traj"]
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    def draw_frame(index):
        """Return None."""
        frame_params = params.copy()
        frame_params["angle_of_attack"] = trial["alpha_traj"][index]
        frame_params["ankle_torque"] = trial["torque_traj"][index]
        model.visualize(trial["state_traj"][:, index], frame_params, ax=ax)
        ax.set_title(f"t = {times[index]:.2f} s")

    # Render at 25 fps while retaining the simulation's smaller timestep.
    fps = 25
    frame_stride = max(1, round(1 / (fps * timestep)))
    frame_indices = list(range(0, times.size, frame_stride))
    if frame_indices[-1] != times.size - 1:
        frame_indices.append(times.size - 1)
    animation = FuncAnimation(
        fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
    )
    animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))
    print(f"Saved {output / 'walker.gif'} ({len(trial['impacts'])} footstrikes).")
    return animation


def plot_roa(roa_data, output):
    converged = roa_data["converged"]
    styles = [("lightgray", "Convergence not confirmed"), ("green", "Converged to upright")]
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.pcolormesh(
        roa_data["theta_values"], roa_data["theta_dot_values"], converged.astype(int),
        cmap=ListedColormap([color for color, _ in styles]), vmin=0, vmax=1,
        shading="nearest",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set(
        xlabel=r"Initial angle $\theta_0$ (rad)",
        ylabel=r"Initial angular velocity $\dot{\theta}_0$ (rad/s)",
        title="Walker: sampled balancing region of attraction",
    )
    ax.legend(
        handles=[Patch(color=color, label=label) for color, label in styles],
        loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
    )
    fig.savefig(output / "roa.png", dpi=300, bbox_inches="tight")

    theta_grid, speed_grid = np.meshgrid(
        roa_data["theta_values"], roa_data["theta_dot_values"]
    )
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for mask, (color, label) in zip((~converged, converged), styles):
        ax.scatter(
            theta_grid[mask], speed_grid[mask], marker="o", s=35, color=color,
            edgecolors="black", linewidths=0.3, label=label, zorder=3,
        )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set(
        xlabel=r"$\theta$ (rad)", ylabel=r"$\dot{\theta}$ (rad/s)",
        xlim=(-0.12, 0.12), ylim=(-0.45, 0.45),
        title="Balancing RoA over passive phase portrait",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)

    phase_theta = np.linspace(-0.12, 0.12, 150)
    phase_speed = np.linspace(-0.45, 0.45, 150)
    theta, speed = np.meshgrid(phase_theta, phase_speed)
    ax.streamplot(
        phase_theta, phase_speed, speed,
        (params["gravity"] / params["length"]) * np.sin(theta),
        color="0.7", density=1.0, linewidth=0.7, arrowsize=0.8, zorder=1,
    )
    fig.savefig(output / "roa_phase.png", dpi=300, bbox_inches="tight")


def plot_return_table(table, output):
    styles = [
        ("captured", "Reached standing RoA", "green"),
        ("returned", "Returned to midstance", "royalblue"),
        ("failed", "Failed trial", "tomato"),
        ("unresolved", "Time limit reached", "lightgray"),
    ]
    codes = np.full(table["outcomes"].shape, np.nan)
    for code, (outcome, _, _) in enumerate(styles):
        codes[table["outcomes"] == outcome] = code

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.pcolormesh(
        table["alpha_values"], table["speed_values"], codes,
        cmap=ListedColormap([color for _, _, color in styles]),
        vmin=-0.5, vmax=len(styles) - 0.5, shading="nearest",
    )
    ax.set(
        xlabel=r"Angle of attack $\alpha$ (rad)",
        ylabel=r"Initial midstance angular velocity $\dot{\theta}_k$ (rad/s)",
        title="Walker: state–action table",
        xlim=(table["alpha_values"][0], table["alpha_values"][-1]),
        ylim=(table["speed_values"][0], table["speed_values"][-1]),
    )
    ax.legend(
        handles=[Patch(color=color, label=label) for _, label, color in styles],
        loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
    )
    fig.savefig(output / "return_table.png", dpi=300, bbox_inches="tight")


def plot_return_map(table, output):
    speeds = table["speed_values"]
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    for column, alpha in enumerate(table["alpha_values"]):
        ax.plot(
            speeds, table["next_speeds"][:, column], marker="o", markersize=4,
            linewidth=1.5, label=rf"$\alpha = {alpha:.4f}$ rad",
        )
    ax.plot(
        speeds, speeds, color="black", linestyle="--", linewidth=1,
        label=r"$\dot{\theta}_{k+1} = \dot{\theta}_k$",
    )
    ax.set(
        xlabel=r"Initial midstance speed $\dot{\theta}_k$ (rad/s)",
        ylabel=r"Next midstance speed $\dot{\theta}_{k+1}$ (rad/s)",
        title="Walker: midstance Poincaré return map", xlim=(0, speeds[-1]),
    )
    ax.set_ylim(bottom=0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output / "poincare_return_map.png", dpi=300, bbox_inches="tight")


def plot_minimum_steps(table, policy, output):
    speeds = table["speed_values"]
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.plot(
        speeds, policy["minimum_steps"], marker="o" if len(speeds) <= 40 else None,
        drawstyle="steps-mid",
    )
    ax.set(
        xlabel="Initial midstance angular velocity (rad/s)",
        ylabel="Predicted minimum number of footstrikes",
        title="Predicted minimum steps to reach the balancing RoA",
    )
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.savefig(output / "minimum_steps.png", dpi=300, bbox_inches="tight")


def plot_walking_trajectory(trial, title, path):
    """Return None."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
    labels = ["Angle (rad)", "Angular velocity (rad/s)"]
    for ax, values, label in zip(axes, trial["state_traj"], labels):
        ax.plot(trial["time_traj"], values)
        ax.set_ylabel(label)
        for impact in trial["impacts"]:
            ax.axvline(impact["time"], color="gray", linestyle="--", alpha=0.5)
        if trial["capture_time"] is not None:
            ax.axvline(trial["capture_time"], color="green", linewidth=2)
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.savefig(path, dpi=300, bbox_inches="tight")


def run_report(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    initial_state = np.array([0.0, args.initial_speed])
    print(f"Generating report with grid {tuple(args.grid)}, timestep {args.timestep:g} s", flush=True)
    roa_data, table = build_control_table(args.grid, args.timestep)
    minimum_policy = poincare.compute_step_policy(table)
    minimum_result = simulate_walking(
        model, initial_state, params, roa_data, table, minimum_policy,
        timestep=args.timestep, sim_time=args.duration,
    )

    maximum_policy = poincare.compute_max_step_policy(table)
    speed_index = poincare._nearest_speed_index(initial_state[1], table["speed_values"])
    maximum_result = None
    if speed_index is None:
        print("The initial speed is outside the return table.")
    elif np.isnan(maximum_policy["maximum_steps"][speed_index]):
        print("No route to the RoA was found for this speed.")
    else:
        predicted = maximum_policy["maximum_steps"][speed_index]
        if np.isposinf(predicted):
            print("The table predicts a repeatable cycle: no finite maximum.")
        else:
            print(f"Table prediction for the maximum-step policy: {predicted:.0f} footstrikes.")
        maximum_result = simulate_walking(
            model, initial_state, params, roa_data, table, maximum_policy,
            timestep=args.timestep, sim_time=10.0,
        )

    summary = {}
    for name, trial, policy in [
        ("minimum", minimum_result, minimum_policy),
        ("maximum", maximum_result, maximum_policy),
    ]:
        if trial is None:
            continue
        predicted = None if speed_index is None else policy[f"{name}_steps"][speed_index]
        observed = steps_before_capture(trial)
        summary[name] = {
            "predicted_steps": predicted, "observed_steps": observed,
            "capture_time": trial["capture_time"], "balanced": bool(roa.has_balanced(trial)),
            "duration": float(trial["time_traj"][-1]),
        }
        print(f"{name.capitalize()} policy: {predicted} predicted; {observed} observed; "
              f"balanced: {summary[name]['balanced']}", flush=True)
    write_json(output / "walking_summary.json", {
        "grid": args.grid, "timestep": args.timestep, "initial_state": initial_state,
        "policies": summary,
    })

    # Measure potential energy relative to the initial stance foot.
    stance_height = 0.0
    for impact in minimum_result["impacts"]:
        theta_minus = impact["state_minus"][0]
        theta_plus = impact["state_plus"][0]
        stance_height += params["length"] * (np.cos(theta_minus) - np.cos(theta_plus))
    energy = (
        model.calculate_energy(minimum_result["state_traj"][:, -1], params)
        + params["mass"] * params["gravity"] * stance_height
    )
    print(f"Final mechanical energy relative to the initial stance foot: {energy:.6g} J")

    if not args.no_animation:
        # Keep the animation alive until the plot windows close.
        _animation = save_walking_animation(minimum_result, args.timestep, output)
    plot_roa(roa_data, output)
    plot_return_table(table, output)
    plot_return_map(table, output)
    plot_minimum_steps(table, minimum_policy, output)

    steps = steps_before_capture(minimum_result)
    title = ("Balancing RoA not reached during this simulation" if steps is None
             else f"Reached the balancing RoA after {steps} footstrikes")
    plot_walking_trajectory(minimum_result, title, output / "walking_trajectory.png")
    if maximum_result is not None:
        steps = steps_before_capture(maximum_result)
        status = (f"No RoA entry; outcome: {maximum_result['outcome']}" if steps is None
                  else f"Reached RoA after {steps} footstrikes")
        plot_walking_trajectory(
            maximum_result, "Maximum-step policy\n" + status,
            output / "maximum_steps_trajectory.png",
        )
    if args.no_show:
        plt.close("all")
    else:
        plt.show()


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    def clean(value):
        """Return a JSON-compatible value with nonfinite numbers replaced by None."""
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [clean(item) for item in value]
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")


def run_diagnostic(args):
    roa_data, table = build_control_table(args.grid, args.timestep)
    compute_policy = (
        poincare.compute_step_policy if args.policy == "minimum"
        else poincare.compute_max_step_policy
    )
    trace, summary = grid_resolution.diagnose_policy(
        model, params, simulate_walking, roa_data, table, compute_policy(table),
        args.initial_speed, args.timestep, trial_time=args.duration,
    )
    print(
        f"Predicted: {summary['predicted_steps']:g} footstrikes; "
        f"observed before capture: {summary['capture_count']:g}; "
        f"balanced: {summary['balanced']}"
    )
    if summary["first_divergence_index"] is not None:
        first = trace[summary["first_divergence_index"]]
        print(
            f"First differing transition: decision {first['decision'] + 1}, "
            f"actual speed {first['actual_speed']:.6f}, grid speed {first['grid_speed']:.6f}, "
            f"alpha {first['alpha']:.6f}. "
            f"Table: {first['predicted_outcome']}; simulation: {first['actual_outcome']}."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = args.output / f"diagnostic_{args.policy}"
    summary.update({
        "grid": args.grid, "policy": args.policy, "params": params,
        "timestep": args.timestep, "trial_time": args.duration,
        "initial_speed": args.initial_speed,
    })
    trace_path = prefix.with_name(prefix.name + "_trace.csv")
    if trace:
        write_csv(trace_path, trace)
    else:
        # An initially captured/failed state can have no foot-placement decisions.
        trace_path.write_text("decision,time,actual_speed,grid_speed,alpha,divergence\n")
    write_json(prefix.with_name(prefix.name + "_summary.json"), summary)
    print(f"Saved diagnostic results to {prefix}_trace.csv and {prefix}_summary.json")


def summarize_grids(results):
    """Return one summary row per grid, compared with the finest grid"""
    rows = []
    reference = results[-1]
    for result in results:
        roa_difference = 0.0
        if "membership" in result:
            roa_difference = 100 * np.mean(result["membership"] != reference["membership"])
        rows.append({
            **dict(zip(["n_theta", "n_theta_dot", "n_speed", "n_alpha"], result["grid"])),
            "balanced": int(result["balanced"].sum()),
            "matched": int(result["matches"].sum()), "total": result["matches"].size,
            "minimum_matched": int(result["matches"][0].sum()),
            "maximum_matched": int(result["matches"][1].sum()),
            "counts_different_or_unverified": int(np.count_nonzero(
                result["observed"] != reference["observed"]
            )),
            "roa_disagreement_percent": float(roa_difference),
        })
    return rows


def run_grid_study(args):
    n_theta, n_theta_dot, n_speed, n_alpha = args.grid
    if args.grids:
        grids = args.grids
    elif args.grid_study == "speed":
        grids = [(n_theta, n_theta_dot, count, n_alpha)
                 for count in (args.levels or [11, 21, 41, 81])]
    elif args.grid_study == "alpha":
        grids = [(n_theta, n_theta_dot, n_speed, count)
                 for count in (args.levels or [5, 9, 17, 33])]
    else:
        grids = [(a, b, n_speed, n_alpha)
                 for a, b in [(9, 13), (17, 25), (33, 49)]]

    if args.grid_study != "combined":
        varying = {"speed": {2}, "alpha": {3}, "roa": {0, 1}}[args.grid_study]
        if any(grid[index] != grids[0][index]
               for grid in grids for index in range(4) if index not in varying):
            raise ValueError("This list changes other grid dimensions; use --grid-study combined.")

    maximum_speed = np.sqrt(2 * params["gravity"] / params["length"])
    # The same probes are used for every candidate; most are between grid rows.
    test_speeds = np.unique(
        args.test_speeds if args.test_speeds is not None
        else np.append(np.linspace(0.0, maximum_speed, 12), args.initial_speed)
    )
    print(f"Study: {args.grid_study}; grids: {grids}", flush=True)
    print(f"Shared timestep: {args.timestep:g} s; {len(test_speeds)} test speeds", flush=True)
    results, details, selected = grid_resolution.check_grid_resolution(
        model, params, simulate_walking, grids, test_speeds,
        balance_timestep=args.timestep, table_timestep=args.timestep,
        walking_timestep=args.timestep, trial_time=args.duration,
    )
    output = args.output / f"grid_{args.grid_study}"
    output.mkdir(parents=True, exist_ok=True)
    trial_columns = [
        "n_theta", "n_theta_dot", "n_speed", "n_alpha", "policy",
        "initial_speed", "predicted_steps", "observed_steps", "outcome", "balanced",
    ]
    write_csv(output / "trials.csv", [dict(zip(trial_columns, row)) for row in details])
    summary = summarize_grids(results)
    for result, row in zip(results, summary):
        print(
            f"{result['grid']}: {row['balanced']}/{row['total']} balanced; "
            f"{row['matched']}/{row['total']} counts matched; "
            f"RoA disagreement {row['roa_disagreement_percent']:.2f}%"
        )
    write_csv(output / "summary.csv", summary)
    write_json(output / "settings.json", {
        "study": args.grid_study, "grids": grids, "test_speeds": test_speeds,
        "params": params, "timestep": args.timestep, "trial_time": args.duration,
        "roa_change_tolerance": 0.01,
        "selected_grid": None if selected is None else results[selected]["grid"],
    })
    print(f"Saved study results to {output}")
    if selected is None:
        print("No grid qualified for selection. Inspect trials.csv; no final grid has been selected.")
    else:
        print("Coarsest tested grid passing this study:", results[selected]["grid"])
        print("Verify the other grid dimensions before using it as the final report grid.")


def run_grid_comparison(args):
    saved = project_root / "output" / "assignment_2"
    folders = args.study_dirs or [
        saved / "grid_speed",
        saved / "refined_speed" / "grid_speed",
        saved / "refined_speed_2" / "grid_speed",
    ]
    results, settings, selected = grid_resolution.load_speed_studies(folders)
    output = args.output / "grid_comparison"
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_grids(results)
    write_csv(output / "summary.csv", summary)
    write_json(output / "settings.json", settings)
    n_theta, n_theta_dot, _, n_alpha = results[0]["grid"]

    notes = [
        (f"The speed study held the RoA grid at {n_theta} × {n_theta_dot}, used {n_alpha} control angles, "
         f"and used a timestep of {settings['timestep']:g} s. Each grid was tested at "
         f"{len(settings['test_speeds'])} initial velocities with both policies."),
        "",
        ("Each walking simulation uses the policy computed on that grid and keeps the continuous state. "
         "The table prediction uses nearest-grid velocities when chaining steps."),
        "",
        ("A velocity grid qualifies when every trial balances with the predicted count, "
         "and every finer tested grid also passes and produces the same observed counts."),
        "",
        "| Velocity samples | Balanced | Predicted counts matched | Counts differing from finest |",
        "| --- | --- | --- | --- |",
    ]
    for row in summary:
        notes.append(f"| {row['n_speed']} | {row['balanced']}/{row['total']} | "
                     f"{row['matched']}/{row['total']} | {row['counts_different_or_unverified']} |")
        print(f"{row['n_speed']} velocities: {row['balanced']}/{row['total']} balanced; "
              f"{row['matched']}/{row['total']} counts matched")
    if selected is None:
        decision = "No grid qualifies yet: matching counts and a finer-grid confirmation are required."
    else:
        grid = results[selected]["grid"]
        decision = (f"{grid[2]} is the coarsest tested velocity resolution meeting the criterion. "
                    "This conclusion applies to these test velocities with the RoA and action grids held fixed.")
    notes.extend(["", decision, ""])
    (output / "notes.md").write_text("\n".join(notes))
    print(decision)
    print(f"Saved summary.csv, settings.json, and notes.md to {output}")


def parse_grid(value):
    """Return four integer grid dimensions, each at least two"""
    try:
        grid = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use four comma-separated integer sizes.") from error
    if len(grid) != 4 or min(grid) < 2:
        raise argparse.ArgumentTypeError("A grid needs four sample counts, each at least 2.")
    return grid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--diagnose", action="store_true", help="Trace one policy; skip plots and animation.")
    mode.add_argument("--grid-study", choices=["speed", "alpha", "roa", "combined"],
                      help="Compare grid sizes; skip plots and animation.")
    mode.add_argument("--plot-grid-comparison", action="store_true",
                      help="Summarize saved speed studies; no simulations are run.")
    parser.add_argument("--study-dirs", type=Path, nargs="+",
                        help="Saved grid_speed folders; defaults to the three existing speed studies.")
    parser.add_argument("--grid", type=int, nargs=4, default=[33, 49, 11, 5],
                        metavar=("ROA_ANGLES", "ROA_SPEEDS", "SPEEDS", "ALPHAS"),
                        help="Report/diagnostic grid, or fixed sizes for a study.")
    parser.add_argument("--levels", type=int, nargs="+", help="Sample counts for a speed or alpha study.")
    parser.add_argument("--grids", type=parse_grid, nargs="+",
                        help="Explicit grids, e.g. 17,25,41,9 33,49,81,17; required for combined.")
    parser.add_argument("--test-speeds", type=float, nargs="+", help="Shared validation velocities (rad/s).")
    parser.add_argument("--initial-speed", type=float, default=3.0)
    parser.add_argument("--policy", choices=["minimum", "maximum"], default="minimum",
                        help="Policy to trace with --diagnose; studies always check both.")
    parser.add_argument("--timestep", type=float, default=1e-4,
                        help="Common RoA, return-table, and walking timestep (seconds).")
    parser.add_argument("--duration", type=float,
                        help="Walking duration: default 10 s for checks, 3 s for report figures.")
    parser.add_argument("--output", type=Path, default=project_root / "output" / "assignment_2")
    parser.add_argument("--no-animation", action="store_true", help="Skip GIF generation in report mode.")
    parser.add_argument("--no-show", action="store_true", help="Save report figures without opening windows.")
    args = parser.parse_args(argv)
    if args.study_dirs and not args.plot_grid_comparison:
        parser.error("--study-dirs requires --plot-grid-comparison.")
    if args.duration is None:
        args.duration = 10.0 if args.diagnose or args.grid_study else 3.0
    if any(count < 2 for count in args.grid):
        parser.error("Every grid dimension must have at least two samples.")
    if not np.isfinite(args.timestep) or args.timestep <= 1e-8:
        parser.error("--timestep must be finite and greater than the 1e-8 s event tolerance.")
    if not np.isfinite(args.duration) or args.duration < 0.5:
        parser.error("--duration must be finite and at least 0.5 s.")
    maximum_speed = np.sqrt(2 * params["gravity"] / params["length"])
    if not np.isfinite(args.initial_speed) or not 0 <= args.initial_speed <= maximum_speed:
        parser.error(f"--initial-speed must be between 0 and {maximum_speed:g} rad/s.")
    if args.levels and (args.grid_study not in ("speed", "alpha") or args.grids):
        parser.error("--levels requires a speed or alpha study without --grids.")
    if args.grids and not args.grid_study:
        parser.error("--grids requires --grid-study.")
    if args.grid_study == "combined" and not args.grids:
        parser.error("A combined study requires an explicit coarse-to-fine --grids list.")
    if args.test_speeds is not None and not args.grid_study:
        parser.error("--test-speeds requires --grid-study.")
    if args.no_show or args.diagnose or args.grid_study or args.plot_grid_comparison:
        plt.switch_backend("Agg")
    if args.plot_grid_comparison:
        try:
            run_grid_comparison(args)
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
    elif args.diagnose:
        run_diagnostic(args)
    elif args.grid_study:
        run_grid_study(args)
    else:
        run_report(args)


if __name__ == "__main__":
    main()
