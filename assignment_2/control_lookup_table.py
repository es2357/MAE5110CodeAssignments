import sys, os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from pathlib import Path
import numpy as np

import simulate_step as simulate


params = {
    "gravity": 9.81,
    "length": 1.0,
    "mass": 1.0,
    "incline": 0.06,
    "angle_of_attack": np.pi / 8,
    "ankle_torque": 0.0,
}


roa_data = np.load("roa_data.npz")

theta_grid = roa_data["theta_grid"]
angular_vel_grid = roa_data["angular_vel_grid"]
roa = roa_data["roa"]

output_dir = Path("resolution_study")
output_dir.mkdir(parents=True, exist_ok=True)

resolutions = [10, 20, 30, 40, 50, 60, 70, 80]

omega_min = 0.0

omega_max = np.sqrt(2 * params["gravity"] / params["length"])

alpha_min = np.pi / 8
alpha_max = np.pi / 7

for resolution in resolutions:

    print(
        f"\nBuilding "
        f"{resolution} x {resolution} "
        f"lookup table..."
    )

    # Create the grid for each resolution desired
    omega_grid = np.linspace(omega_min, omega_max, resolution)
    alpha_grid = np.linspace(alpha_min, alpha_max, resolution)

    # Calculate adjacent grid spacing
    delta_omega = (omega_grid[1] - omega_grid[0])
    delta_alpha = (alpha_grid[1] - alpha_grid[0])

    # Create arrays to hold the possible outcomes for each pair
    next_omega_table = np.full((len(omega_grid), len(alpha_grid)), np.nan)
    reaches_roa = np.zeros((len(omega_grid), len(alpha_grid)), dtype=bool)
    failed_table = np.zeros((len(omega_grid), len(alpha_grid)), dtype=bool)
    already_in_roa = np.zeros(len(omega_grid), dtype=bool)

    for i, omega in enumerate(omega_grid):
        for j, alpha in enumerate(alpha_grid):

            status, result = simulate.simulate_step(
                omega,
                alpha,
                theta_grid,
                angular_vel_grid,
                roa,
                params
            )

            if status == "already in roa":
                already_in_roa[i] = True
                break

            elif status == "roa":
                reaches_roa[i, j] = True

            elif status == "poincare":
                next_omega_table[i, j] = result

            elif status == "failed":
                failed_table[i, j] = True
            else:
                raise ValueError(f"unexpected status from simulate_step(): {status}")

    steps_to_roa = np.full(len(omega_grid), -1, dtype=int)
    control_alpha = np.full(len(omega_grid), np.nan)

    for i in range(len(omega_grid)):
        if already_in_roa[i]:
            steps_to_roa[i] = 0

    target_steps = 1
    
    while True:
        newly_resolved = 0
        for i in range(len(omega_grid)):

            if steps_to_roa[i] != -1:
                continue

            for j in range(len(alpha_grid)):
                if target_steps == 1:
                    if reaches_roa[i, j]:
                        steps_to_roa[i] = 1
                        control_alpha[i] = (alpha_grid[j])
                        newly_resolved += 1
                        break

                else:
                    next_omega = (next_omega_table[i, j])

                    if np.isnan(next_omega):
                        continue

                    if (
                        next_omega < omega_grid[0] or
                        next_omega > omega_grid[-1]
                    ):
                        continue

                    next_i = np.argmin(np.abs(omega_grid - next_omega))

                    if (
                        steps_to_roa[next_i]
                        == target_steps - 1
                    ):
                        steps_to_roa[i] = (target_steps)
                        control_alpha[i] = (alpha_grid[j])
                        newly_resolved += 1
                        break
        print(f"Step depth {target_steps}: {newly_resolved} new states")
        if newly_resolved == 0:
            break
        target_steps += 1

    poincare_transitions = np.sum(~np.isnan(next_omega_table))
    roa_transitions = np.sum(reaches_roa)
    failed_transitions = np.sum(failed_table)
    unresolved_states = np.sum(steps_to_roa == -1)

    filename = (output_dir/ f"lookup_{resolution}x{resolution}.npz")

    np.savez_compressed(
        filename,

        resolution=resolution,

        omega_grid=omega_grid,
        alpha_grid=alpha_grid,

        delta_omega=delta_omega,
        delta_alpha=delta_alpha,

        next_omega_table=next_omega_table,
        reaches_roa=reaches_roa,
        failed_table=failed_table,

        already_in_roa=already_in_roa,
        steps_to_roa=steps_to_roa,
        control_alpha=control_alpha,

        omega_min=omega_min,
        omega_max=omega_max,
        alpha_min=alpha_min,
        alpha_max=alpha_max,

        poincare_transitions=poincare_transitions,
        roa_transitions=roa_transitions,
        failed_transitions=failed_transitions,
        unresolved_states=unresolved_states,
    )


    print(f"Saved {filename}")
    print(f"Delta omega = "f"{delta_omega:.5f} rad/s")

    print(
        f"Maximum nearest-neighbor "
        f"omega error = "
        f"{delta_omega / 2:.5f} rad/s"
    )

    print(
        f"Delta alpha = "
        f"{delta_alpha:.6f} rad "
        f"({np.degrees(delta_alpha):.4f} deg)"
    )

    print(
        f"Unresolved states = "
        f"{unresolved_states}"
    )

print("\nFinished all resolution tables.")