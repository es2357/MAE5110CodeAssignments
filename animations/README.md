# Rimless-wheel animations

This folder contains the animation scripts, supporting models, generated GIFs,
physics tests, and dependencies. It has its own `pyproject.toml`, `uv.lock`, and
`.venv`, and can run independently of the homework project.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
these commands from `MAE5110CodeAssignments`:

```console
cd animations
uv sync --python 3.14
```

This installs NumPy, SciPy, Matplotlib, Pillow, and the pytest development
dependency into this folder's `.venv`. Use `animations/.venv/bin/python` as the
interpreter when opening these scripts from the assignment project in an editor.
All commands below run from `animations/` unless stated otherwise.

## Animated phase plot and impact energy

```console
uv run python rimless_wheel_analysis.py
```

This saves `rimless_wheel_phase_energy.gif` and a PNG preview of the first impact
alongside the script, then opens the live animation. The default run shows five
downhill impacts, starting at an angular speed of 1.5 rad/s. Four synchronized
views show the phase trajectories and energy contours, wheel motion, energy
history, and PE/KE/reset equations with changing numerical values.

To export without opening a window, use `--no-show`; for a live animation without
saving files, use `--no-save`:

```console
uv run python rimless_wheel_analysis.py --no-show
uv run python rimless_wheel_analysis.py --no-save
```

From `MAE5110CodeAssignments`, select this animation project explicitly:

```console
uv run --project animations python animations/rimless_wheel_analysis.py
```

From the parent `legged_robots` directory, the equivalent command is:

```console
uv run --project MAE5110CodeAssignments/animations python MAE5110CodeAssignments/animations/rimless_wheel_analysis.py
```

The phase contours use $H=K+mg\ell\cos\theta$, where
$K=\tfrac12m\ell^2\dot\theta^2$. Physical energy uses a fixed height reference:
$PE=mg(y_f+\ell\cos\theta)$ and $E=K+PE$, with $y_f$ the stance foot height.
Both $H$ and $E$ stay constant within a stance. The foot reference changes at
impact, while the hub height and physical PE remain continuous. The reset gives
$K^+=K^-\cos^2(2\alpha)$, so the lost energy is
$K^-\sin^2(2\alpha)$: 50% of incoming KE for the default eight spokes.
See the [MIT rimless-wheel notes](https://underactuated.mit.edu/simple_legs.html#section2)
for the stance dynamics and impact map.

Playback is slowed for explanation and pauses around each impact. The dashed
phase-space arrow illustrates an instantaneous coordinate reset; it does not
represent physical motion through the intermediate states. The energy plot
uses physical simulation time, with an abrupt drop at each impact.

Adjust physical parameters in
[`animation_models/rimless_wheel.py`](animation_models/rimless_wheel.py)'s
`generate_params()`. Animation options include `--steps`, `--initial-speed`,
`--samples-per-step`, `--fps`, `--dpi`, and `--output` (a `.gif` path; the PNG uses
the same stem). Relative `--output` paths are resolved from the directory where
you launch the command. For example:

```console
uv run python rimless_wheel_analysis.py --steps 6 --initial-speed 1.6 --fps 20 --dpi 120 --output rimless_demo.gif --no-show
```

## Three branches of the return map

```console
uv run python rimless_wheel_return_map_animation.py --no-show
```

This saves `rimless_wheel_return_map_cases.gif` and a PNG preview alongside the
script. Three panels show the first collision for each branch, using the default
Earth gravity, eight spokes, unit length and mass, and gamma = 0.08 rad:

- **A, +1.5 rad/s:** crosses upright and steps downhill.
- **B, +0.9 rad/s:** stops before upright, reverses under gravity, and collides
  again at its starting contact angle.
- **C, -2.3 rad/s:** crosses upright in reverse and steps uphill.

The signed one-impact map starts positive velocities at theta = gamma-alpha
and negative velocities at theta = gamma+alpha. Thus C uses the opposite
starting contact from A and B. The shared plot shows the three analytic
branches, the identity line, and each example's input/output pair. Hollow
markers show predicted outputs; filled markers show simulated impact outputs.
Colors distinguish the branches, rather than eventual regions of attraction.

Playback pauses at upright (A/C), reversal (B), and either side of impact. The
panels align these events for comparison and display independent physical
clocks. Impact changes the stance foot and angular velocity instantaneously;
the wheel's world position stays continuous. Each example ends after its first
collision, so the GIF illustrates the return map rather than long-term motion.

Remove `--no-show` to also open the animation. Use `--no-save` for a live preview
without exporting; `--fps`, `--dpi`, and `--output example.gif` adjust the export.
Frame counts in `make_frames()` control playback and pauses, and the commented
axis limits in `create_animation()` control the visible physical area.
The example velocities are set in `animation_models/rimless_wheel_branches.py`;
it checks that parameter changes preserve the intended branch for each case.

## One-step wheel animation

```console
uv run python rimless_wheel_animation.py
```

This shows one stance and the spoke handoff. After you close the live animation
window, it saves `rimless_wheel_one_step.gif` alongside the script. Its physical
parameters and frame counts are set in the script's `Parameters` section.

## Physics checks

Check energy conservation during stance, hub and PE continuity at impact,
collision losses, and the analytic return map with:

```console
uv run python -m pytest tests/test_rimless_wheel_energy.py -q
uv run python -m pytest tests/test_rimless_wheel_branches.py -q
```
