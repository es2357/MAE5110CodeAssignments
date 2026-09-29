from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import simulate_step as simulate
import stabilizing_upright as controller


base_dir = Path(__file__).resolve().parent

params = {
    "gravity": 9.81,
    "length": 1.0,
    "mass": 1.0,
    "incline": 0.06,
    "angle_of_attack": np.pi / 8,
    "ankle_torque": 0.0,
}

# Selected grid resolution from study conducted
resolution = 50

roa_data = np.load(base_dir / "roa_data.npz")
theta_grid = roa_data["theta_grid"]
angular_vel_grid = roa_data["angular_vel_grid"]
roa = roa_data["roa"]


lookup = np.load(
    base_dir
    / "resolution_study"
    / f"lookup_{resolution}x{resolution}.npz"
)

omega_grid = lookup["omega_grid"]
alpha_grid = lookup["alpha_grid"]
next_omega_table = lookup["next_omega_table"]
reaches_roa = lookup["reaches_roa"]
already_in_roa = lookup["already_in_roa"]
steps_to_roa = lookup["steps_to_roa"]
control_alpha = lookup["control_alpha"]


# Discrete lookup-table prediction to see if a state has a path that enter the roa in a exact # of steps
max_horizon = 20

n_omega = len(omega_grid)
n_alpha = len(alpha_grid)


exact_reachable = np.zeros((max_horizon + 1,n_omega),dtype=bool)
exact_action = np.full((max_horizon + 1, n_omega), -1, dtype=int)
exact_reachable[0, :] = (already_in_roa)


for h in range(1, max_horizon + 1):
    for i in range(n_omega):

        if already_in_roa[i]:
            continue

        for j in range(n_alpha):
            # If it takes exactly one step to reach roa
            if h == 1:
                if reaches_roa[i, j]:
                    exact_reachable[h, i] = True
                    exact_action[h,i] = j

                    break
            else:
                next_omega = (next_omega_table[i, j])
                if np.isnan(next_omega):
                    continue

                if (next_omega < omega_grid[0] or next_omega > omega_grid[-1]):
                    continue

                next_i = np.argmin(np.abs(omega_grid - next_omega))

                if exact_reachable[h - 1, next_i]:
                    exact_reachable[h, i] = True
                    exact_action[h, i] = j
                    break

# Max path to reach roa
max_steps_to_roa = np.full(n_omega, -1, dtype=int)

for i in range(n_omega):

    possible_steps = np.where(exact_reachable[:, i])[0]
    if possible_steps.size > 0:
        max_steps_to_roa[i] = int(possible_steps.max())


def append_segment(all_time, all_states, segment_time,segment_states):
    if len(all_time) == 0:
        all_time.extend(segment_time.tolist())
        all_states.extend(segment_states.tolist())
    else:
        offset = all_time[-1]
        all_time.extend((offset + segment_time[1:]).tolist())
        all_states.extend(segment_states[1:].tolist())



# Minimum step lookup rollout policy
def rollout_minimum(initial_omega):

    current_omega = float(initial_omega)

    all_time = []
    all_states = []

    poincare_omegas = [current_omega]
    alpha_history = []
    actual_steps = 0

    for _ in range(max_horizon):
        current_state = np.array([0.0, current_omega])

        if controller.is_valid_capture(
            current_state,
            theta_grid,
            angular_vel_grid,
            roa,
            params
        ):
            return (
                np.asarray(all_time),
                np.asarray(all_states),
                np.asarray(poincare_omegas),
                np.asarray(alpha_history),
                actual_steps,
                True
            )

        current_i = np.argmin(np.abs(omega_grid - current_omega))

        if (steps_to_roa[current_i] == 0):
            print(
                "WARNING: nearest lookup node claims the state is in the RoA, but the actual continuous state failed validation test")

            return (
                np.asarray(all_time),
                np.asarray(all_states),
                np.asarray(poincare_omegas),
                np.asarray(alpha_history),
                actual_steps,
                False
            )

        if (steps_to_roa[current_i] < 0):
            print("Minimum-step policy became unresolved during the continuous rollout.")

            return (
                np.asarray(all_time),
                np.asarray(all_states),
                np.asarray(poincare_omegas),
                np.asarray(alpha_history),
                actual_steps,
                False
            )

        alpha = control_alpha[current_i]

        if np.isnan(alpha):
            print(
                "No minimum-step action is available.")

            return (
                np.asarray(all_time),
                np.asarray(all_states),
                np.asarray(poincare_omegas),
                np.asarray(alpha_history),
                actual_steps,
                False
            )

        alpha_history.append(alpha)

        status, result, segment_time, segment_states = simulate.simulate_step_history(
            current_omega,
            alpha,
            theta_grid,
            angular_vel_grid,
            roa,
            params
        )

        append_segment(all_time, all_states, segment_time, segment_states)

        if (status == "already in roa"):
            return (
                np.asarray(all_time),
                np.asarray(all_states),
                np.asarray(poincare_omegas),
                np.asarray(alpha_history),
                actual_steps,
                True
            )

        if status in ("roa", "poincare"):
            actual_steps += 1

        if status == "roa":
            return (
                np.asarray(all_time),
                np.asarray(all_states),
                np.asarray(poincare_omegas),
                np.asarray(alpha_history),
                actual_steps,
                True
            )


        if status == "poincare":
            current_omega = float(result)
            poincare_omegas.append(current_omega)
        else:
            print(
                "Minimum-step rollout failed "
                f"with status: {status}"
            )

            return (
                np.asarray(all_time),
                np.asarray(all_states),
                np.asarray(poincare_omegas),
                np.asarray(alpha_history),
                actual_steps,
                False
            )

    print("Minimum-step rollout reached the search horizon.")

    return (
        np.asarray(all_time),
        np.asarray(all_states),
        np.asarray(poincare_omegas),
        np.asarray(alpha_history),
        actual_steps,
        False
    )



def rollout_exact_steps(initial_omega, number_of_steps):

    current_omega = float(initial_omega)
    remaining = int(number_of_steps)
    
    actual_steps = 0

    all_time = []
    all_states = []

    poincare_omegas = [current_omega]
    alpha_history = []


    while remaining > 0:

        current_state = np.array([0.0, current_omega])

        if controller.is_valid_capture(
            current_state,
            theta_grid,
            angular_vel_grid,
            roa,
            params
        ):
            print("Actual continuous trajectory entered the RoA earlier than predicted.")

            return (
                np.asarray(all_time),
                np.asarray(all_states),
                np.asarray(poincare_omegas),
                np.asarray(alpha_history),
                actual_steps,
                False
            )


        current_i = np.argmin(np.abs(omega_grid - current_omega))
        j = exact_action[remaining, current_i]

        if j < 0:
            print(
                "No exact-step action found "
                f"with {remaining} steps remaining."
            )

            return (
                np.asarray(all_time),
                np.asarray(all_states),
                np.asarray(poincare_omegas),
                np.asarray(alpha_history),
                actual_steps,
                False
            )

        alpha = alpha_grid[j]
        alpha_history.append(alpha)

        status, result, segment_time, segment_states = simulate.simulate_step_history(current_omega, alpha, theta_grid, angular_vel_grid,roa,params)

        append_segment(all_time, all_states, segment_time, segment_states)


        if (status == "already in roa"):
            print(
                "Actual trajectory was already captured before the expected walking step.")

            return (
                np.asarray(all_time),
                np.asarray(all_states),
                np.asarray(poincare_omegas),
                np.asarray(alpha_history),
                actual_steps,
                False
            )

        if status in ("roa","poincare"):
            actual_steps += 1

        if status == "roa":
            success = (remaining == 1 and actual_steps == number_of_steps)

            if not success:
                print(
                    "Actual rollout entered the RoA earlier than the lookup path predicted.")

            return (
                np.asarray(all_time),
                np.asarray(all_states),
                np.asarray(poincare_omegas),
                np.asarray(alpha_history),
                actual_steps,
                success
            )

        if status == "poincare":
            current_omega = float(result)
            poincare_omegas.append(current_omega)
            remaining -= 1
        else:
            print(
                "Exact-step rollout failed "
                f"with status: {status}"
            )

            return (
                np.asarray(all_time),
                np.asarray(all_states),
                np.asarray(poincare_omegas),
                np.asarray(alpha_history),
                actual_steps,
                False
            )

    print(
        "Requested number of steps was completed, but the actual trajectory did not enter the RoA."
    )

    return (
        np.asarray(all_time),
        np.asarray(all_states),
        np.asarray(poincare_omegas),
        np.asarray(alpha_history),
        actual_steps,
        False
    )


# Select a walking state that the lookup table predicts requires 3 or more steps.
three_step_candidates = np.where(steps_to_roa == 3)[0]
longer_candidates = np.where(steps_to_roa > 3)[0]

candidate_indices = []


if (three_step_candidates.size > 0):

    middle = (len(three_step_candidates) // 2)
    middle_candidate = (three_step_candidates[middle])
    candidate_indices.append(middle_candidate)

    for candidate_i in (three_step_candidates):
        if (candidate_i != middle_candidate):
            candidate_indices.append(candidate_i)

for candidate_i in (longer_candidates):
    candidate_indices.append(candidate_i)

if len(candidate_indices) == 0:
    raise RuntimeError("No lookup state is predicted to require at least 3 steps.")

initial_i = None
initial_omega = None
minimum_result = None

for candidate_i in (candidate_indices):
    candidate_omega = float(omega_grid[candidate_i])

    trial_result = (rollout_minimum(candidate_omega))
    trial_steps = (trial_result[4])
    trial_success = (trial_result[5])

    if (trial_success and trial_steps >= 3):
        initial_i = (candidate_i)
        initial_omega = (candidate_omega)

        minimum_result = (trial_result)

        break


if initial_i is None:
    raise RuntimeError("No continuously validated initial condition was found that requires at least 3 walking steps.")

# Find minimum step candidate and print those values
min_time, min_states, min_poincare, min_alpha, min_actual_steps, min_success = minimum_result

print(
    "\nChosen initial omega = "
    f"{initial_omega:.4f} rad/s"
)

print(
    "Lookup-table predicted minimum steps = "
    f"{steps_to_roa[initial_i]}"
)

print(
    "Actual minimum-policy rollout = "
    f"{min_actual_steps} steps"
)


# Find maximum step candidate and print those values
candidate_max_steps = int(max_steps_to_roa[initial_i])

print(
    "Longest discrete candidate path = "
    f"{candidate_max_steps} steps"
)

if (candidate_max_steps == max_horizon):
    print("WARNING: longest path reached the search horizon. A longer path may exist.")


validated_max_steps = -1

max_time = np.array([])
max_states = np.array([])
max_poincare = np.array([])
max_alpha = np.array([])


if candidate_max_steps > 0:

    for candidate_steps in range(candidate_max_steps, 0,-1):
        print(
            "\nTesting continuous "
            f"{candidate_steps}-step "
            "maximum-path candidate..."
        )

        trial_time, trial_states, trial_poincare, trial_alpha, trial_actual_steps, trial_success = rollout_exact_steps(initial_omega, candidate_steps)

        if (trial_success and trial_actual_steps == candidate_steps):
            validated_max_steps = (candidate_steps)
            max_time = (trial_time)
            max_states = (trial_states)
            max_poincare = (trial_poincare)
            max_alpha = (trial_alpha)

            break

# Print results
print(
    "\nMinimum-step lookup-policy rollout = "
    f"{min_actual_steps} steps"
)

if validated_max_steps >= 0:
    print(
        "Longest validated lookup-policy rollout found = "
        f"{validated_max_steps} steps"
    )
else:
    print(
        "No exact-length maximum-policy candidate was validated continuously."
    )


# Plot phase portrait
theta_mesh, omega_mesh = np.meshgrid(theta_grid, angular_vel_grid)

plt.figure(figsize=(9, 6))

# Standing RoA
plt.contourf(
    theta_mesh,
    omega_mesh,
    roa.astype(int),
    levels=[0.5, 1.5],
    alpha=0.15
)

# Minimum-step lookup-policy trajectory
if min_states.size > 0:
    plt.plot(
        min_states[:, 0],
        min_states[:, 1],
        label=(
            "Minimum-step lookup-policy "
            f"trajectory ({min_actual_steps} steps)"
        )
    )


# Longest continuously validated trajectory
if (validated_max_steps >= 0 and max_states.size > 0):
    plt.plot(
        max_states[:, 0],
        max_states[:, 1],
        label=(
            "Longest validated lookup-policy "
            f"trajectory ({validated_max_steps} steps)"
        )
    )


# Minimum-policy Poincare crossings
if min_poincare.size > 0:
    plt.scatter(
        np.zeros(len(min_poincare)),
        min_poincare,
        marker="o",
        label=(
            "Minimum-policy "
            "Poincaré crossings"
        )
    )

# Maximum-policy Poincare crossings
if (validated_max_steps >= 0 and max_poincare.size > 0):
    plt.scatter(
        np.zeros(len(max_poincare)),
        max_poincare,
        marker="x",
        label=(
            "Maximum-policy "
            "Poincaré crossings"
        )
    )

# RoA legend patch
roa_patch = Patch(
    alpha=0.15,
    label=(
        "Standing Region "
        "of Attraction"
    )
)

handles, labels = (plt.gca().get_legend_handles_labels())
handles.append(roa_patch)

labels.append("Standing Region of Attraction")

plt.xlabel(r"$\theta$ [rad]")
plt.ylabel(r"$\dot{\theta}$ [rad/s]")

plt.title(
    "Walking Trajectories from "
    rf"$\dot{{\theta}}_0="
    f"{initial_omega:.3f}$ rad/s"
)

plt.legend(handles, labels)
plt.grid()
plt.tight_layout()
plt.show()


valid = (steps_to_roa >= 0)

if np.any(valid):
    plt.figure(figsize=(8, 5))

    plt.step(omega_grid[valid], steps_to_roa[valid], where="mid")
    plt.scatter(omega_grid[valid], steps_to_roa[valid])

    plt.xlabel(
        r"Initial Poincaré Velocity "
        r"$\dot{\theta}_0$ [rad/s]"
    )

    plt.ylabel("Predicted Walking Steps Before Entering RoA")
    plt.title("Predicted Steps Required to Reach Standstill Region")
    plt.yticks(np.arange(0,np.max(steps_to_roa[valid]) + 1))

    plt.grid()
    plt.tight_layout()
    plt.show()

else:
    print("No lookup states have a valid steps-to-RoA value.")