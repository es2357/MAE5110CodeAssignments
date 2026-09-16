# MAE 5110 Code Assignments

Code assignments for MAE 5110.

## Installation

Install [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/). After cloning this repository, run the following command from its root directory:

```console
uv sync --python 3.14
```

This creates a local `.venv` and installs the required dependencies. Run Python commands inside the environment with `uv run`, for example:

```console
uv run python assignment_0/assignment_0.py
uv run python assignment_0/bouncing_ball_sim.py
```

The Assignment 0 scripts locate the shared `models` and `integrators` folders
from their own file paths, so they also work when launched from inside
`assignment_0/`. Use the assignment project's `.venv/bin/python` in your editor.
From the parent `legged_robots` directory, select the project explicitly:

```console
uv run --project MAE5110CodeAssignments python MAE5110CodeAssignments/assignment_0/assignment_0.py
uv run --project MAE5110CodeAssignments python MAE5110CodeAssignments/assignment_0/bouncing_ball_sim.py
```

The independent [animations project](animations/README.md) contains the
rimless-wheel animations, generated GIFs, and their setup and physics notes.

## Assignment 1 scripts and results

The Assignment 1 scripts live in `assignment_1/`. From the
`MAE5110CodeAssignments` project root, run:

```console
uv run python assignment_1/assignment_1.py
uv run python assignment_1/roa_sim.py
uv run python assignment_1/roa_plot.py --no-show
uv run python assignment_1/rimless_wheel_return_map.py --no-show
uv run python assignment_1/floquet.py --gamma 0.10 --spokes 8 --no-show
uv run python assignment_1/plot_summary.py --no-show
```

From the parent `legged_robots` workspace root, select the assignment project
explicitly and include its directory in the script path, for example:

```console
uv run --project MAE5110CodeAssignments python MAE5110CodeAssignments/assignment_1/floquet.py --gamma 0.10 --spokes 8 --no-show
```

The analysis scripts save their CSV and PNG results in
`assignment_1/assignment_1_data/`, regardless of the directory from which you
run them. The folder is created automatically. RoA CSV and PNG filenames include
the slope and spoke count, such as `rimless_wheel_roa_gamma_0.100_N_8.csv` and
`rimless_wheel_roa_gamma_0.100_N_8.png`. By default, `roa_plot.py` reads the CSV
named by its `DEFAULT_CSV` constant.
The plot title and contact-angle boundaries use the parameters saved in the CSV.
To select another saved result, pass its filename:

```console
uv run python assignment_1/roa_plot.py rimless_wheel_roa_gamma_0.100_N_8.csv --no-show
```

Run that command from the project root after simulating gamma 0.100 with 8 spokes. You can
also pass an explicit CSV path; its plot is saved in
`assignment_1/assignment_1_data/`.

`plot_summary.py` saves four comparison figures in `assignment_1/assignment_1_data/`:

- `rimless_wheel_energy_loss_summary.png`: percentage of kinetic energy lost per impact.
- `rimless_wheel_incline_summary.png`: incline versus rolling RoA fraction and Floquet multiplier, for 6 through 12 spokes.
- `rimless_wheel_spoke_summary.png`: spoke count versus the same quantities, at inclines of 0.08, 0.16, and 0.26 rad.
- `rimless_wheel_convergence_time_summary.png`: local convergence per second, accounting for the time between impacts.

The first run uses `summary_sim.py` to fill missing cases and saves the results
to CSV. Later runs reuse these grids. Use `--workers 1` to run simulations
sequentially, or keep the default of three workers. The sweep includes
45 supported combinations of N = 6 through 12 and gamma = 0.04, 0.08, 0.12,
0.16, 0.20, 0.26, and 0.39 rad. Cases with gamma greater than or equal to pi/N
are omitted because the current settling and return-map conditions assume the
shallower-slope regime.

Each summary case uses 21 angles across its contact interval and 41 speeds
from -3 to 3 rad/s. Matching saved 81 by 161 grids are downsampled to this same
grid; new cases use the existing simulator with a 0.01 s timestep and impact
time refinement to 1e-9 s. Unclassified runs are retried with 60 s instead of
30 s. The plots report the percentage of **all sampled states** that reach
rolling, with unresolved outcomes kept separately. Since the angle interval
changes with N, this is a comparison of sampled fractions, not absolute basin
areas. The summary CSV records the settings, source files, and differences
from the available denser grids. New files start with `summary_roa_` so the
original full-grid results are preserved.

Floquet estimates use simulated returns near each existing rolling fixed point.
Missing cycles have no multiplier. Smaller multipliers mean faster local
convergence per impact; they depend on spoke count rather than incline in this
model. The time comparison uses `-ln(abs(lambda))/T`, where T is the simulated
steady step duration: larger values mean faster local decay per second. This
describes small disturbances near steady rolling, not the time for arbitrary
starting states to reach an attractor. Aggregated values are saved in
`rimless_wheel_parameter_summary.csv`.

## Assignments

- [Assignment 0](assignments/assignment_0.md)
- [Assignment 1](assignments/assignment_1.md)
- [Assignment 2](assignments/assignment_2.md)