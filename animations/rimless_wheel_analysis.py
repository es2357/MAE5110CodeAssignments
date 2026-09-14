"""Animate phase contours, stance motion, and impact energy loss.

Run ``uv run python rimless_wheel_analysis.py`` to save a GIF and show the
animation, or add ``--no-show`` for an export without a window.
Equations: https://underactuated.mit.edu/simple_legs.html#section2
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from PIL import Image

from animation_models import rimless_wheel
from animation_models.rimless_wheel_energy import simulate

INK = "#202b3a"
MUTED = "#667586"
BLUE = "#1976b5"
GOLD = "#c28212"
TEAL = "#087e80"
PINK = "#bf386d"
BACKGROUND = "#f7f8fa"
STEP_COLORS = ["#2576b7", "#168b91", "#6556a5", "#bc7429", "#417d56"]


@dataclass(frozen=True)
class Frame:
    step: int
    sample: int
    stage: str = "flow"
    progress: float = 0.0


class CompactGifWriter(PillowWriter):
    """Store indexed frames to reduce memory use during a long GIF export."""

    def grab_frame(self, **savefig_kwargs):
        super().grab_frame(**savefig_kwargs)
        self._frames[-1] = (
            self._frames[-1]
            .convert("RGB")
            .quantize(
                colors=128,
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.NONE,
            )
        )


def make_frames(stances):
    frames = [Frame(0, 0, "start")] * 16
    for step, stance in enumerate(stances):
        last = len(stance.time) - 1
        frames.extend(Frame(step, i) for i in range(last + 1))
        frames.extend([Frame(step, last, "before")] * 8)
        # Only the illustrative arrow moves here; physical time and state freeze.
        frames.extend(
            Frame(step, last, "reset", float(p)) for p in np.linspace(0, 1, 10)
        )
        frames.extend([Frame(step, last, "after")] * 14)
    frames.extend([Frame(len(stances) - 1, len(stances[-1].time) - 1, "end")] * 32)
    return frames


def style_axis(ax):
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b8c3cd")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color="#e9edf1", linewidth=0.7)
    ax.set_axisbelow(True)


def create_animation(stances, params, fps=20):
    """Return (figure, live animation, frame descriptors, update callback)."""
    mass, length, gravity = (params[k] for k in ("mass", "length", "gravity"))
    spoke_count = int(params["num_spokes"])
    alpha = np.pi / spoke_count
    gamma = params["slope_angle"]
    lower, upper = rimless_wheel.contact_angles(params)
    retention = np.cos(2 * alpha) ** 2
    frames = make_frames(stances)
    colors = [STEP_COLORS[i % len(STEP_COLORS)] for i in range(len(stances))]
    loss_before = np.r_[0, np.cumsum([s.loss for s in stances])]

    # Duplicate physical timestamps at resets produce true vertical energy drops.
    history = {key: [] for key in ("time", "kinetic", "potential", "energy")}
    history_starts = []
    for stance in stances:
        history_starts.append(len(history["time"]))
        for key, values in history.items():
            values.extend(getattr(stance, key))
            after = (
                stance.time[-1] if key == "time" else getattr(stance, "after_" + key)
            )
            values.append(after)
    history = {key: np.asarray(value) for key, value in history.items()}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titleweight": "bold",
            "font.size": 10,
            "savefig.facecolor": BACKGROUND,
        }
    )
    fig = plt.figure(figsize=(13, 8.6), facecolor=BACKGROUND)
    fig.text(
        0.06,
        0.96,
        "RIMLESS WHEEL  /  Motion, contours & impact loss",
        fontsize=20,
        weight="bold",
    )
    fig.text(
        0.06,
        0.924,
        rf"{params['num_spokes']} spokes     $\alpha=\pi/{params['num_spokes']}$"
        rf"     $\gamma={gamma:.2f}$ rad     $m={mass:g}$ kg"
        rf"     $\ell={length:g}$ m     {len(stances)} downhill impacts",
        fontsize=10,
        color=MUTED,
    )

    phase = fig.add_axes((0.065, 0.505, 0.55, 0.335))
    wheel = fig.add_axes((0.68, 0.555, 0.28, 0.275))
    energy = fig.add_axes((0.065, 0.10, 0.55, 0.215))
    for ax in (phase, energy):
        style_axis(ax)

    phase.set_title("01  PHASE SPACE", loc="left", fontsize=12, pad=34)
    phase.text(
        0,
        1.035,
        r"Thin curves: constant $H=K+mg\ell\cos\theta$ (J)",
        transform=phase.transAxes,
        color=MUTED,
        fontsize=9,
    )
    xmax = upper + 0.16
    xmin = lower - 0.16
    wmax = max(np.max(s.omega) for s in stances) * 1.18
    theta_grid = np.linspace(xmin, xmax, 400)
    omega_grid = np.linspace(-wmax, wmax, 360)
    theta_mesh, omega_mesh = np.meshgrid(theta_grid, omega_grid)
    contour_energy = (
        0.5 * mass * length** 2 * omega_mesh** 2
        + mass * gravity * length * np.cos(theta_mesh)
    )
    levels = (
        mass
        * gravity
        * length
        * np.array([0.90, 0.95, 1.0, 1.025, 1.06, 1.10, 1.16, 1.23])
    )
    contours = phase.contour(
        theta_mesh,
        omega_mesh,
        contour_energy,
        levels=levels,
        colors="#b6c2ce",
        linewidths=0.8,
    )
    phase.clabel(contours, inline=True, fontsize=7, fmt="%.2f")
    phase.axvspan(xmin, lower, color="#e8ebef", alpha=0.8)
    phase.axvspan(upper, xmax, color="#e8ebef", alpha=0.8)
    phase.axvline(lower, color=PINK, alpha=0.7, linewidth=1.5)
    phase.axvline(upper, color=PINK, alpha=0.7, linewidth=1.5)
    phase.axhline(0, color="#8e9baa", linewidth=0.8)
    phase.axvline(0, color="#8e9baa", linewidth=0.8, linestyle=":")
    phase.set(
        xlim=(xmin, xmax),
        ylim=(-wmax, wmax),
        xlabel=r"$\theta$ (rad)",
        ylabel=r"$\dot\theta$ (rad/s)",
    )
    phase.set_xticks(
        [lower, 0, upper],
        [
            r"$\gamma-\alpha$" + f"\n{lower:.3f}",
            "0",
            r"$\gamma+\alpha$" + f"\n{upper:.3f}",
        ],
    )
    phase.text(
        0.5,
        0.045,
        "Physical stance angles lie between the pink boundaries",
        ha="center",
        transform=phase.transAxes,
        fontsize=8,
        color=MUTED,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )

    (contour_top,) = phase.plot([], [], color=TEAL, linewidth=1.1, alpha=0.65)
    (contour_bottom,) = phase.plot([], [], color=TEAL, linewidth=1.1, alpha=0.65)
    trails = [phase.plot([], [], color=color, linewidth=2.5)[0] for color in colors]
    reset_paths = [
        phase.plot([], [], "--", color=PINK, linewidth=1.25, alpha=0.55)[0]
        for _ in stances
    ]
    (current_dot,) = phase.plot(
        [],
        [],
        "o",
        color=INK,
        markersize=7,
        zorder=8,
        markeredgecolor="white",
        markeredgewidth=1.5,
    )
    (impact_minus,) = phase.plot(
        [], [], "o", color=PINK, fillstyle="none", markersize=8, zorder=7
    )
    (impact_plus,) = phase.plot(
        [], [], "s", color=PINK, fillstyle="none", markersize=7, zorder=7
    )
    reset_arrow = phase.annotate(
        "",
        xy=(0, 0),
        xytext=(0, 0),
        arrowprops={
            "arrowstyle": "-|>",
            "color": PINK,
            "linewidth": 2,
            "linestyle": "--",
        },
        zorder=6,
    )
    phase.legend(
        handles=[
            Line2D([], [], color=BLUE, lw=2.5, label="Stance trajectory"),
            Line2D([], [], color=PINK, lw=1.5, ls="--", label="Instantaneous reset"),
        ],
        loc="lower left",
        bbox_to_anchor=(0, 0.14),
        fontsize=8,
        framealpha=0.95,
    )
    phase_status = fig.text(0.065, 0.411, "", fontsize=10, weight="bold", color=TEAL)
    phase_detail = fig.text(0.065, 0.386, "", fontsize=9, color=MUTED)

    wheel.set_title("02  THE SAME MOTION", loc="left", fontsize=12, pad=32)
    wheel.set_aspect("equal")
    wheel.set_axis_off()
    (ground,) = wheel.plot([], [], color=INK, linewidth=2)
    spokes = [
        wheel.plot([], [], color="#b5bec8", lw=2)[0]
        for _ in range(spoke_count)
    ]
    (hub_marker,) = wheel.plot([], [], "o", color=INK, markersize=8)
    (foot_marker,) = wheel.plot([], [], "o", color=BLUE, markersize=7)
    (next_marker,) = wheel.plot([], [], "o", color=GOLD, markersize=5)
    (vertical,) = wheel.plot([], [], "--", color=MUTED, linewidth=1)
    wheel_status = fig.text(0.68, 0.525, "", fontsize=10, weight="bold")
    fig.text(
        0.68, 0.501, "Blue: stance spoke    Gold: next contact", fontsize=9, color=MUTED
    )

    fig.text(
        0.065,
        0.356,
        "03  ENERGY FROM A FIXED HEIGHT REFERENCE",
        fontsize=12,
        weight="bold",
    )
    energy.text(
        0,
        1.035,
        "Slowed playback; time pauses for resets. Energy drops are instantaneous.",
        transform=energy.transAxes,
        fontsize=8.5,
        color=MUTED,
    )
    energy.set(
        xlim=(-0.04, history["time"][-1] * 1.05),
        ylim=(
            min(0, history["potential"].min()) - 0.35,
            history["energy"].max() * 1.09,
        ),
        xlabel="Physical time (s)",
        ylabel="Energy (J)",
    )
    energy_lines = {}
    for name, color, label in (
        ("potential", GOLD, "PE"),
        ("kinetic", BLUE, "KE"),
        ("energy", TEAL, "E = PE + KE"),
    ):
        (energy_lines[name],) = energy.plot(
            [], [], color=color, linewidth=2.4, label=label
        )
    energy.legend(
        loc="center left", bbox_to_anchor=(0.02, 0.53), fontsize=8, framealpha=0.95
    )
    time_cursor = energy.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")
    drop_lines = []
    drop_labels = []
    for i, stance in enumerate(stances):
        (line,) = energy.plot(
            [stance.time[-1]] * 2,
            [stance.energy[-1], stance.after_energy],
            color=PINK,
            linewidth=3,
            visible=False,
        )
        label = energy.annotate(
            f"−{stance.loss:.2f}",
            (stance.time[-1], stance.energy[-1]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=PINK,
            visible=False,
        )
        drop_lines.append(line)
        drop_labels.append(label)

    fig.text(0.68, 0.457, "04  ENERGY & RESET EQUATIONS", fontsize=12, weight="bold")
    kinetic_text = fig.text(0.68, 0.418, "", color=BLUE, fontsize=11)
    potential_text = fig.text(0.68, 0.382, "", color=GOLD, fontsize=11)
    total_text = fig.text(0.68, 0.346, "", color=TEAL, fontsize=11)
    local_text = fig.text(0.68, 0.310, "", fontsize=10)
    fig.text(
        0.68,
        0.284,
        r"$y_f$: stance-foot height; fixed datum $y=0$.",
        fontsize=8.5,
        color=MUTED,
    )
    fig.text(
        0.68,
        0.245,
        r"$\theta^+=\theta^- -2\alpha,\quad"
        r"\dot\theta^+=\dot\theta^-\cos(2\alpha)$",
        fontsize=11,
        color=PINK,
    )
    fig.text(
        0.68,
        0.211,
        rf"$K^+=K^-\cos^2(2\alpha)={retention:.2f}K^-$",
        fontsize=11,
        color=PINK,
    )
    impact_text = fig.text(
        0.68, 0.181, "", fontsize=10, color=PINK, linespacing=1.6, va="top"
    )
    budget_text = fig.text(0.68, 0.079, "", fontsize=10, weight="bold")
    fig.text(
        0.065,
        0.031,
        "PE stays continuous at impact. Changing stance foot shifts the local contour reference: "
        r"$H=E-mgy_f$.",
        fontsize=9,
        color=MUTED,
    )

    def update(index):
        frame = frames[index]
        stance = stances[frame.step]
        after = frame.stage in ("after", "end")
        theta, omega = (
            stance.after_state
            if after
            else (stance.theta[frame.sample], stance.omega[frame.sample])
        )
        foot = stance.after_foot if after else stance.foot
        kinetic = stance.after_kinetic if after else stance.kinetic[frame.sample]
        potential = stance.after_potential if after else stance.potential[frame.sample]
        total = stance.after_energy if after else stance.energy[frame.sample]
        local = total - mass * gravity * foot[1]
        current_time = stance.time[frame.sample]
        losses = loss_before[frame.step + int(after)]
        impact = frame.stage in ("before", "reset", "after", "end")
        active_color = colors[frame.step]

        speed_squared = (
            2
            * (local - mass * gravity * length * np.cos(theta_grid))
            / (mass * length**2)
        )
        contour_speed = np.sqrt(np.maximum(speed_squared, 0))
        contour_speed[speed_squared < 0] = np.nan
        contour_top.set_data(theta_grid, contour_speed)
        contour_bottom.set_data(theta_grid, -contour_speed)
        for line in (contour_top, contour_bottom):
            line.set_color(active_color)
        for i, previous in enumerate(stances):
            visible_samples = (
                len(previous.time)
                if i < frame.step or (i == frame.step and after)
                else frame.sample + 1
                if i == frame.step
                else 0
            )
            trails[i].set_data(
                previous.theta[:visible_samples], previous.omega[:visible_samples]
            )
            trails[i].set_alpha(1 if i == frame.step else 0.55)
            reset_visible = i < frame.step or (i == frame.step and after)
            reset_paths[i].set_data(
                [previous.theta[-1], previous.after_state[0]] if reset_visible else [],
                [previous.omega[-1], previous.after_state[1]] if reset_visible else [],
            )
            drop_lines[i].set_visible(reset_visible)
            drop_labels[i].set_visible(reset_visible)
        current_dot.set_data([theta], [omega])
        impact_minus.set_data(
            [stance.theta[-1]] if impact else [], [stance.omega[-1]] if impact else []
        )
        impact_plus.set_data(
            [stance.after_state[0]] if after else [],
            [stance.after_state[1]] if after else [],
        )
        reset_arrow.set_visible(frame.stage == "reset")
        if frame.stage == "reset":
            before = np.array([stance.theta[-1], stance.omega[-1]])
            reset_arrow.xy = before + frame.progress * (stance.after_state - before)
            reset_arrow.set_position(before)

        hub = foot + length * np.array([np.sin(theta), np.cos(theta)])
        angles = theta + 2 * alpha * np.arange(spoke_count)
        endpoints = hub - length * np.column_stack((np.sin(angles), np.cos(angles)))
        for k, line in enumerate(spokes):
            line.set_data([hub[0], endpoints[k, 0]], [hub[1], endpoints[k, 1]])
            line.set_color(
                BLUE if k == 0 else GOLD if k == len(spokes) - 1 else "#b5bec8"
            )
            line.set_linewidth(3 if k in (0, len(spokes) - 1) else 1.5)
        ground_x = hub[0] + length * np.array([-1.5, 1.5])
        ground.set_data(ground_x, -np.tan(gamma) * ground_x)
        hub_marker.set_data([hub[0]], [hub[1]])
        foot_marker.set_data([foot[0]], [foot[1]])
        next_marker.set_data([endpoints[-1, 0]], [endpoints[-1, 1]])
        vertical.set_data([foot[0], foot[0]], [foot[1], foot[1] + 1.3 * length])
        wheel.set_xlim(hub[0] - 1.5 * length, hub[0] + 1.5 * length)
        wheel.set_ylim(hub[1] - 1.35 * length, hub[1] + 1.15 * length)
        wheel_status.set_text(
            f"Step {frame.step + 1}/{len(stances)}  ·  t = {current_time:.3f} s"
            + ("  ·  IMPACT" if impact else "")
        )
        wheel_status.set_color(PINK if impact else INK)

        upto = history_starts[frame.step] + (
            len(stance.time) + 1 if after else frame.sample + 1
        )
        for name, line in energy_lines.items():
            line.set_data(history["time"][:upto], history[name][:upto])
        time_cursor.set_xdata([current_time] * 2)
        kinetic_text.set_text(
            r"$K=\frac{1}{2}m\ell^2\dot\theta^2$" + f" = {kinetic:.3f} J"
        )
        potential_text.set_text(
            r"$PE=mg(y_f+\ell\cos\theta)$" + f" = {potential:.3f} J"
        )
        total_text.set_text(r"$E=K+PE$" + f" = {total:.3f} J")
        local_text.set_text(r"$H=K+mg\ell\cos\theta$" + f" = {local:.3f} J")
        budget_text.set_text(f"Cumulative impact loss: {losses:.3f} J")
        if impact:
            phase_status.set_color(PINK)
            phase_status.set_text(
                f"Impact {frame.step + 1}: "
                + (
                    "reset complete"
                    if after
                    else "instantaneous reset; physical time is frozen"
                )
            )
            phase_detail.set_text(
                rf"$(\theta^-,\dot\theta^-) = ({stance.theta[-1]:.3f}, {stance.omega[-1]:.3f})$"
                + r"  $\longrightarrow$  "
                + rf"$(\theta^+,\dot\theta^+) = ({stance.after_state[0]:.3f}, {stance.after_state[1]:.3f})$"
            )
            impact_text.set_text(
                f"KE: {stance.kinetic[-1]:.3f} → {stance.after_kinetic:.3f} J\n"
                + rf"$E^- - E^+ = K^-\sin^2(2\alpha)$ = {stance.loss:.3f} J"
                + "\nPE is unchanged at contact."
            )
        else:
            phase_status.set_color(active_color)
            exchange = (
                "Climbing: PE rises, KE falls"
                if theta < 0
                else "Descending: PE falls, KE rises"
            )
            if abs(theta) < 0.022:
                exchange = "At upright: PE is maximal, KE is minimal for this stance"
            phase_status.set_text(exchange)
            phase_detail.set_text(
                f"Step {frame.step + 1} follows H = {local:.3f} J.  "
                "Both H and E stay constant during stance."
            )
            impact_text.set_text(
                "During stance: PE ↔ KE\n"
                + r"$\dot E=\dot H=0,\quad\ddot\theta=(g/\ell)\sin\theta$"
                + "\nImpact removes KE at the next contact."
            )
        return []

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=False,
        repeat=True,
        repeat_delay=1000,
        cache_frame_data=False,
    )
    update(0)
    return fig, animation, frames, update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", type=int, default=5, help="Number of downhill impacts (default: 5)"
    )
    parser.add_argument(
        "--initial-speed",
        type=float,
        default=1.5,
        help="Postimpact initial speed in rad/s",
    )
    parser.add_argument(
        "--samples-per-step", type=int, default=48, help="Animation samples per stance"
    )
    parser.add_argument(
        "--fps", type=int, default=20, help="Playback frames per second"
    )
    parser.add_argument(
        "--dpi", type=int, default=100, help="GIF and preview resolution"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("rimless_wheel_phase_energy.gif"),
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Save files without opening a window"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Show live animation without exporting files",
    )
    args = parser.parse_args()
    if args.fps <= 0 or args.dpi <= 0:
        parser.error("--fps and --dpi must be positive")
    if args.no_show:
        plt.switch_backend("Agg")
    params = rimless_wheel.generate_params()
    try:
        stances = simulate(
            params,
            steps=args.steps,
            initial_speed=args.initial_speed,
            samples_per_step=args.samples_per_step,
        )
    except ValueError as exc:
        parser.error(str(exc))
    fig, animation, frames, update = create_animation(stances, params, fps=args.fps)

    if not args.no_save:
        if args.output.suffix.lower() != ".gif":
            parser.error("--output must end in .gif")
        args.output.parent.mkdir(parents=True, exist_ok=True)

        def progress(current, total):
            if current % 80 == 0 or current == total - 1:
                print(f"Rendering frame {current + 1}/{total}", flush=True)

        animation.save(
            args.output,
            writer=CompactGifWriter(fps=args.fps),
            dpi=args.dpi,
            progress_callback=progress,
        )
        # A still of the first completed impact is useful for notes and previews.
        first_impact = next(
            i for i, frame in enumerate(frames) if frame.stage == "after"
        )
        update(first_impact)
        preview = args.output.with_suffix(".png")
        fig.savefig(preview, dpi=args.dpi)
        print(f"Saved GIF: {args.output.resolve()}")
        print(f"Saved preview: {preview.resolve()}")
    print(
        f"Impacts: {len(stances)}; cumulative KE loss: {sum(s.loss for s in stances):.6f} J"
    )
    print(
        f"Each impact retains {np.cos(2 * np.pi / params['num_spokes']) ** 2:.0%} of incoming KE."
    )
    if not args.no_show:
        update(0)
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
