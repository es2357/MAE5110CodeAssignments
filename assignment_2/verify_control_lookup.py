import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import simulate_step as simulate
import stabilizing_upright as controller



base_dir = Path(__file__).resolve().parent

resolution_dir = base_dir / "resolution_study"
resolution_dir.mkdir(parents=True, exist_ok=True)


params = {
    "gravity": 9.81,
    "length": 1.0,
    "mass": 1.0,
    "incline": 0.06,
    "angle_of_attack": np.pi / 8,
    "ankle_torque": 0.0,
}


# Load saved file data
roa_data = np.load(base_dir / "roa_data.npz")

theta_grid = roa_data["theta_grid"]
angular_vel_grid = roa_data["angular_vel_grid"]
roa = roa_data["roa"]

resolutions = [10, 20, 30, 40, 50, 60, 70, 80,]


omega_min = 0.0
omega_max = np.sqrt(2 * params["gravity"] / params["length"])

alpha_min = np.pi / 8
alpha_max = np.pi / 7


# Create off-grid inital conditions for the policy-rollout validation
n_validation_omega = 50
n_validation_alpha = 50


omega_edges = np.linspace(omega_min, omega_max, n_validation_omega + 1)
validation_omegas = 0.5 * (omega_edges[:-1] + omega_edges[1:])

alpha_edges = np.linspace(alpha_min, alpha_max,n_validation_alpha + 1)
validation_alphas = 0.5 * (alpha_edges[:-1] + alpha_edges[1:])

validation_pairs = []

for omega in validation_omegas:
    for alpha in validation_alphas:
        validation_pairs.append([omega,alpha])

validation_pairs = np.asarray(validation_pairs)

# Reference simulation results
print("\nComputing direct validation transitions...")

true_status = []

true_next_omega = np.full(len(validation_pairs), np.nan)

for k in range(len(validation_pairs)):

    omega = validation_pairs[k, 0]
    alpha = validation_pairs[k, 1]

    status, result = (
        simulate.simulate_step(
            omega,
            alpha,
            theta_grid,
            angular_vel_grid,
            roa,
            params
        )
    )

    true_status.append(status)

    if status == "poincare":
        true_next_omega[k] = (result)

true_status = np.asarray(true_status, dtype=str)


def lookup_prediction(omega, alpha, lookup_data):
    omega_grid = lookup_data["omega_grid"]
    alpha_grid = lookup_data["alpha_grid"]
    next_omega_table = lookup_data["next_omega_table"]
    reaches_roa = lookup_data["reaches_roa"]
    failed_table = lookup_data["failed_table"]
    already_in_roa = lookup_data["already_in_roa"]

    if (
        omega < omega_grid[0]
        or
        omega > omega_grid[-1]
        or
        alpha < alpha_grid[0]
        or
        alpha > alpha_grid[-1]
    ):
        return ("outside", np.nan)

    # Nearest neighbor approximation to find state
    i = np.argmin(np.abs(omega_grid - omega))
    j = np.argmin(np.abs(alpha_grid - alpha))

    # Determine the transition type
    if already_in_roa[i]:
        return ("already in roa",np.nan)

    if reaches_roa[i, j]:
        return ("roa", np.nan)

    if not np.isnan(next_omega_table[i, j]):
        return ("poincare", next_omega_table[i, j])

    if failed_table[i, j]:
        return ("failed", np.nan)

    return ("unknown",np.nan)


# Use off-grid states to simulate and nearest neighbor finds control input alpha.
# That control input is then used for the next walking step.
def rollout_policy(initial_omega, lookup_data, theta_grid, angular_vel_grid, roa, params, max_steps=30):
    omega_grid = lookup_data["omega_grid"]
    steps_to_roa = lookup_data["steps_to_roa"]
    control_alpha = lookup_data["control_alpha"]

    current_omega = float(initial_omega)

    # Lookup-table prediction using nearest-neighbor approx.
    initial_i = np.argmin(np.abs(omega_grid - current_omega))
    predicted_steps = int(steps_to_roa[initial_i])

    if predicted_steps < 0:
        return (False, predicted_steps, -1, "unresolved initial state")

    steps_taken = 0


    # Use selected policy until convergence or failure
    while steps_taken <= max_steps:

        # Poincare state
        current_state = np.array([0.0,current_omega])

        # Conservative RoA rule
        if controller.is_valid_capture(current_state, theta_grid, angular_vel_grid, roa, params):
            return (True, predicted_steps, steps_taken, "captured")

        # Nearest neighbor for policy action
        current_i = np.argmin(np.abs(omega_grid - current_omega))

        # Guard if lookup says requires zero steps but the full capture test fails
        if (steps_to_roa[current_i] == 0):
            return (False, predicted_steps, steps_taken, "false capture")

        # No policy exists for this state
        if (steps_to_roa[current_i] < 0):
            return (False, predicted_steps, steps_taken, "unresolved successor")

        alpha = control_alpha[current_i]

        if np.isnan(alpha):
            return (False, predicted_steps, steps_taken, "no action")

        # Max horizon guard
        if steps_taken >= max_steps:
            return (False, predicted_steps, steps_taken, "horizon")

        status, result = (
            simulate.simulate_step(
                current_omega,
                alpha,
                theta_grid,
                angular_vel_grid,
                roa,
                params
            )
        )

        if (status == "already in roa"):
            return (True, predicted_steps, steps_taken, "captured")

        steps_taken += 1

        if status == "roa":
            return (True, predicted_steps, steps_taken,"captured")

        if status == "poincare":
            current_omega = float(result)
        else:
            return (False, predicted_steps, steps_taken, status)

    return (False, predicted_steps, steps_taken, "horizon")


# Store results
summary = []
all_predictions = {}

# Test every resolutions lookup-table
for resolution in resolutions:

    print(
        f"\nValidating "
        f"{resolution} x {resolution} "
        f"lookup table..."
    )

    lookup_file = (
        resolution_dir
        /
        f"lookup_{resolution}x"
        f"{resolution}.npz"
    )

    lookup_data = np.load(lookup_file)

    omega_grid = lookup_data["omega_grid"]
    alpha_grid = lookup_data["alpha_grid"]
    steps_to_roa = lookup_data["steps_to_roa"]
    control_alpha = lookup_data["control_alpha"]

    delta_omega = float(lookup_data["delta_omega"])
    delta_alpha = float(lookup_data["delta_alpha"])


    predicted_status = []

    predicted_next_omega = np.full(len(validation_pairs), np.nan)

    omega_lookup_error = np.zeros(len(validation_pairs))
    alpha_lookup_error = np.zeros(len(validation_pairs))
    

    for k in range(len(validation_pairs)):

        omega = validation_pairs[k, 0]
        alpha = validation_pairs[k, 1]

        i = np.argmin(np.abs(omega_grid - omega))
        j = np.argmin(np.abs(alpha_grid - alpha))

        omega_lookup_error[k] = abs(omega - omega_grid[i])
        alpha_lookup_error[k] = abs(alpha - alpha_grid[j])

        status, next_omega = lookup_prediction( omega, alpha, lookup_data)

        predicted_status.append(status)
        predicted_next_omega[k] = (next_omega)

    predicted_status = np.asarray(predicted_status, dtype=str)

    # Grid metric classification errors between transitions
    classification_mismatch = (predicted_status != true_status)
    classification_mismatch_count = int(np.sum(classification_mismatch))
    classification_mismatch_rate = (classification_mismatch_count / len(validation_pairs))

    # Transition map Poincare error
    comparable = (
        (true_status == "poincare")
        &
        (predicted_status == "poincare")
    )

    if np.any(comparable):
        transition_error = np.abs(predicted_next_omega[comparable] - true_next_omega[comparable])

        mean_transition_error = float(np.mean(transition_error))
        max_transition_error = float(np.max(transition_error))
        rms_transition_error = float(np.sqrt(np.mean(transition_error ** 2)))
        number_comparable = int(np.sum(comparable))
    else:
        mean_transition_error = (np.nan)
        max_transition_error = (np.nan)
        rms_transition_error = (np.nan)
        number_comparable = 0

    # Calculate nearest neighbor error
    max_omega_lookup_error = float(np.max(omega_lookup_error))
    mean_omega_lookup_error = float(np.mean(omega_lookup_error))
    max_alpha_lookup_error = float(np.max(alpha_lookup_error))
    mean_alpha_lookup_error = float(np.mean(alpha_lookup_error))


    policy_success = []
    policy_predicted_steps = []
    policy_actual_steps = []
    policy_reasons = []

    for omega0 in validation_omegas:
        success, predicted_steps, actual_steps, reason = rollout_policy(omega0, lookup_data, theta_grid, angular_vel_grid, roa, params)

        policy_success.append(success)
        policy_predicted_steps.append(predicted_steps)
        policy_actual_steps.append(actual_steps)
        policy_reasons.append(reason)

    policy_success = np.asarray(policy_success, dtype=bool)
    policy_predicted_steps = (np.asarray(policy_predicted_steps, dtype=int))
    policy_actual_steps = (np.asarray(policy_actual_steps, dtype=int))


    resolved_policy = (policy_predicted_steps >= 0)
    policy_resolved_count = int(np.sum(resolved_policy))
    policy_unresolved_count = int(np.sum(~resolved_policy))
    
    successful_resolved = (resolved_policy & policy_success)
    policy_success_count = int(np.sum(successful_resolved))


    if policy_resolved_count > 0:
        policy_success_rate = (policy_success_count / policy_resolved_count)
    else:
        policy_success_rate = (np.nan)


    # Predicted # of steps vs actual
    step_comparable = (successful_resolved & (policy_actual_steps >= 0))


    if np.any(step_comparable):
        policy_step_error = np.abs(policy_actual_steps[step_comparable] - policy_predicted_steps[step_comparable])
        policy_exact_match_count = int(np.sum(policy_step_error == 0))
        policy_exact_match_rate = (policy_exact_match_count / np.sum(step_comparable))
        mean_policy_step_error = float(np.mean(policy_step_error))
        max_policy_step_error = int(np.max(policy_step_error))
    else:
        policy_exact_match_count = 0
        policy_exact_match_rate = (np.nan)
        mean_policy_step_error = (np.nan)
        max_policy_step_error = (np.nan)

    false_capture_count = int(sum(reason == "false capture" for reason in policy_reasons))


    # Save summary for specified resolution
    summary.append(
        [
            resolution,

            delta_omega,
            delta_alpha,

            max_omega_lookup_error,
            mean_omega_lookup_error,

            max_alpha_lookup_error,
            mean_alpha_lookup_error,

            classification_mismatch_count,
            classification_mismatch_rate,

            number_comparable,

            mean_transition_error,
            max_transition_error,
            rms_transition_error,

            policy_resolved_count,
            policy_unresolved_count,

            policy_success_count,
            policy_success_rate,

            policy_exact_match_count,
            policy_exact_match_rate,

            mean_policy_step_error,
            max_policy_step_error,

            false_capture_count,
        ]
    )

    all_predictions[resolution] = {"status": predicted_status, "next_omega": predicted_next_omega}


    # Print for debugging/results
    print(
        f"Classification mismatches: "
        f"{classification_mismatch_count} "
        f"/ {len(validation_pairs)}"
    )

    print(
        f"Mean transition error: "
        f"{mean_transition_error:.6f} rad/s"
    )

    print(
        f"RMS transition error: "
        f"{rms_transition_error:.6f} rad/s"
    )

    print(
        f"Maximum transition error: "
        f"{max_transition_error:.6f} rad/s"
    )

    print(
        f"Policy resolved states: "
        f"{policy_resolved_count}"
    )

    print(
        f"Complete rollout successes: "
        f"{policy_success_count}"
        f"/{policy_resolved_count}"
    )

    print(
        f"Complete rollout success rate: "
        f"{100 * policy_success_rate:.2f}%"
    )

    print(
        f"Exact step-count match rate: "
        f"{100 * policy_exact_match_rate:.2f}%"
    )

    print(
        f"Mean step-count error: "
        f"{mean_policy_step_error:.3f}"
    )

    print(
        f"Maximum step-count error: "
        f"{max_policy_step_error}"
    )

    print(
        f"False captures: "
        f"{false_capture_count}"
    )


# Grid refinement comparison
refinement_results = []

for k in range(len(resolutions) - 1):
    coarse_resolution = (resolutions[k])
    fine_resolution = (resolutions[k + 1])

    coarse_status = (all_predictions[coarse_resolution]["status"])
    fine_status = (all_predictions[fine_resolution]["status"])

    coarse_next_omega = (all_predictions[coarse_resolution]["next_omega"])
    fine_next_omega = (all_predictions[fine_resolution]["next_omega"])

    status_disagreements = int(np.sum(coarse_status != fine_status))

    comparable = (
        (coarse_status == "poincare")
        &
        (fine_status == "poincare")
    )

    if np.any(comparable):
        transition_change = np.abs(coarse_next_omega[comparable] - fine_next_omega[ comparable])
        mean_transition_change = float(np.mean(transition_change))
        max_transition_change = float(np.max(transition_change))
    else:
        mean_transition_change = (np.nan)
        max_transition_change = (np.nan)

    refinement_results.append(
        [
            coarse_resolution,
            fine_resolution,
            status_disagreements,
            mean_transition_change,
            max_transition_change,
        ]
    )


# Save results
summary_array = np.asarray(summary, dtype=float)

np.savetxt(
    resolution_dir
    / "grid_resolution_summary.csv",

    summary_array,

    delimiter=",",

    header=(
        "resolution,"
        "delta_omega,"
        "delta_alpha,"
        "max_omega_lookup_error,"
        "mean_omega_lookup_error,"
        "max_alpha_lookup_error,"
        "mean_alpha_lookup_error,"
        "classification_mismatch_count,"
        "classification_mismatch_rate,"
        "number_comparable,"
        "mean_transition_error,"
        "max_transition_error,"
        "rms_transition_error,"
        "policy_resolved_count,"
        "policy_unresolved_count,"
        "policy_success_count,"
        "policy_success_rate,"
        "policy_exact_match_count,"
        "policy_exact_match_rate,"
        "mean_policy_step_error,"
        "max_policy_step_error,"
        "false_capture_count"
    ),

    comments=""
)


refinement_array = np.asarray(refinement_results, dtype=float)

np.savetxt(
    resolution_dir
    / "grid_refinement_summary.csv",

    refinement_array,

    delimiter=",",

    header=(
        "coarse_resolution,"
        "fine_resolution,"
        "status_disagreements,"
        "mean_transition_change,"
        "max_transition_change"
    ),

    comments=""
)

# Data for plotting
resolution_values = (summary_array[:, 0])
mean_transition_errors = (summary_array[:, 10])
max_transition_errors = (summary_array[:, 11])
rms_transition_errors = (summary_array[:, 12])
policy_success_rates = (summary_array[:, 16])
policy_exact_match_rates = (summary_array[:, 18])
mean_policy_step_errors = (summary_array[:, 19])
max_policy_step_errors = (summary_array[:, 20])



# Figure 1 plotting
plt.figure(figsize=(8, 5))

plt.plot(
    resolution_values,
    mean_transition_errors,
    marker="o",
    label="Mean Error"
)

plt.plot(
    resolution_values,
    rms_transition_errors,
    marker="o",
    label="RMS Error"
)

plt.plot(
    resolution_values,
    max_transition_errors,
    marker="o",
    label="Maximum Error"
)

plt.xlabel("Lookup Table Resolution")
plt.ylabel(r"Error in $\dot{\theta}_{k+1}$ [rad/s]")
plt.title("Poincaré Transition-Map Error vs. Grid Resolution")

plt.xticks(resolution_values)

plt.grid()
plt.legend()
plt.tight_layout()

plt.show()


# Figure 2 transition class changes
refinement_labels = [
    f"{int(row[0])}"
    f"\N{RIGHTWARDS ARROW}"
    f"{int(row[1])}"

    for row
    in refinement_results
]

classification_changes = (refinement_array[:, 2])

plt.figure(figsize=(8, 5))

plt.plot(
    refinement_labels,
    classification_changes,
    marker="o"
)

plt.xlabel("Successive Grid Refinement")
plt.ylabel("Number of Transition Classification Changes")
plt.title("Sensitivity of Transition Classification to Grid Refinement")

plt.grid()
plt.tight_layout()


plt.show()


