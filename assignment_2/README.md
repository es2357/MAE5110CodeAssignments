# Running Assignment 2

Instructions to run assingment 2 can be found below:

Run all commands from the repository root, `MAE5110CodeAssignments`

## To run assingment 2

```bash
uv run python assignment_2/assignment_2.py
```

This builds the RoA and lookup table, simulates both walking policies from an initial angle of 0 rad and angular velocity of 3 rad/s, and generates the plots and `walker.gif`. Results are saved in `output/assignment_2/`, and the plot windows open when generation finishes.

The defaults use a 33 by 49 RoA grid, an 11-velocity by 5-angle lookup table, and a timestep of 0.0001 s. The minimum-policy trajectory runs for 3 seconds, while the maximum-policy trajectory runs for 10 seconds. To reproduce the final report's finer grid and longer minimum-policy trajectory, use the command below.

## To generate the final simulation results and plots

```bash
uv run python assignment_2/assignment_2.py \
  --grid 33 49 641 5 \
  --timestep 0.001 \
  --duration 10 \
  --initial-speed 3 \
  --output output/assignment_2/final_grid \
  --no-animation --no-show
```

This estimates the region of attraction (RoA), builds the lookup table, computes both policies, and simulates walking from an initial angle of 0 rad and angular velocity of 3 rad/s for 10 seconds.

The four numbers after `--grid` specify:

| Number | Meaning |
| --- | --- |
| 33 | Angle samples for the RoA |
| 49 | Angular-velocity samples for the RoA |
| 641 | Midstance velocities in the lookup table |
| 5 | Foot-placement angles in the lookup table |

Results are saved in `output/assignment_2/final_grid/`, including:

- `walking_trajectory.png`: the minimum-policy trajectory.
- `maximum_steps_trajectory.png`: the maximum-policy trajectory.
- `minimum_steps.png`: predicted footsteps versus initial velocity.
- `return_table.png`: outcomes for the sampled velocities and foot-placement angles.
- `poincare_return_map.png`: return velocities for the sampled foot-placement angles.
- `roa.png`: the estimated region of attraction.
- `walking_summary.json`: predicted and observed footstep counts, and whether each trajectory balances.

For these settings, the expected result is **3 footstrikes for the minimum policy and 5 for the maximum policy**, with both trajectories reaching standing. These counts refer to footstrikes before entering the RoA.

Remove `--no-animation` to also generate `walker.gif`. Remove `--no-show` to open the plot windows. Running the command again regenerates the files in the specified output folder.

## To generate the return-table comparison picture

```bash
uv run python assignment_2/plot_return_tables.py
```

This compares tables with **11, 321, and 641 velocity samples**, keeping the RoA grid and five foot-placement angles fixed. It saves the comparison to:

```text
output/assignment_2/grid_comparison/return_table_resolution.png
```

The script reuses the saved 641-row table when its settings match. Otherwise, it builds and saves the table first. The coarser tables use the corresponding rows from that table.

## Optional: to rerun the numerical grid check

To reproduce the grid-resolution numbers in the report, run:

```bash
uv run python assignment_2/assignment_2.py \
  --grid-study speed \
  --grid 33 49 321 5 \
  --levels 321 641 1281 \
  --timestep 0.001 \
  --duration 10 \
  --output output/assignment_2/grid_verification
```

This takes longer because it builds all three tables and checks **26 walking trials per grid**: 13 initial velocities with each of the two policies. It compares the table's predicted footstep count with the simulated count and checks whether the walker reaches standing.

Results are saved in `output/assignment_2/grid_verification/grid_speed/`:

- `summary.csv`: the overall results for each grid.
- `trials.csv`: predictions and simulation outcomes for each initial velocity and policy.
- `settings.json`: the settings used and the selected grid, if one qualifies.

The expected results are:

| Velocity samples | Trials that balance with the predicted count |
| --- | --- |
| 321 | 25/26 |
| 641 | 26/26 |
| 1281 | 26/26 |

The 641-sample grid is the coarsest tested velocity grid that passes, with the 1281-sample grid confirming the same observed counts. This conclusion applies to the tested initial velocities with the RoA grid and foot-placement angles held fixed.
