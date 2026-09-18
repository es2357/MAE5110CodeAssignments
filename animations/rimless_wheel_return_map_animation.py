"""Export a GIF showing the physical motions behind the three return-map branches.

Run from this folder with:
    uv run python rimless_wheel_return_map_animation.py --no-show

Each panel stops immediately after its first collision. Playback aligns key
events for comparison; the displayed physical clocks run independently.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

from animation_models.rimless_wheel import generate_params
from animation_models.rimless_wheel_branches import simulate_cases
from rimless_wheel_analysis import CompactGifWriter, style_axis

INK = "#202b3a"
MUTED = "#667586"
BACKGROUND = "#f7f8fa"
COLORS = {"A": "#30EBC9", "B": "#EA4259", "C": "#D55E00"}
THRESHOLD_COLOR = "#CC1278"
TITLES = {
    "A": "A   FORWARD STEP",
    "B": "B   TURN BACK",
    "C": "C   BACKWARD STEP",
}
FORMULAS = {
    "A": r"$P(w)=c\sqrt{w^2+D}$",
    "B": r"$P(w)=-cw$",
    "C": r"$P(w)=-c\sqrt{w^2-D}$",
}


class ReturnMapGifWriter(CompactGifWriter):
    """Keep thin moving arrows intact across frames with different palettes."""

    def finish(self):
        self._frames[0].save(
            self.outfile,
            save_all=True,
            append_images=self._frames[1:],
            duration=int(1000 / self.fps),
            loop=0,
            disposal=2,
            optimize=False,
        )


@dataclass(frozen=True)
class Frame:
    stage: str
    progress: float = 0.0


def make_frames():
    """Change these counts to adjust playback speed and explanatory pauses."""
    frames = [Frame("start")] * 16
    frames += [Frame("approach", p) for p in np.linspace(0, 1, 40)]
    frames += [Frame("critical")] * 24
    frames += [Frame("depart", p) for p in np.linspace(0, 1, 40)]
    frames += [Frame("before")] * 16
    # The impact is instantaneous: no interpolation between pre/post states.
    frames += [Frame("after")] * 36
    return frames


def frame_state(motion, frame):
    """Return physical time, theta, omega and stance foot for a display frame."""
    critical_index = motion.turn_index if motion.key == "B" else motion.upright_index
    critical_time = motion.time[critical_index]
    if frame.stage == "after":
        return motion.time[-1], *motion.after_state, motion.after_foot
    if frame.stage == "start":
        time = 0.0
    elif frame.stage == "approach":
        time = frame.progress * critical_time
    elif frame.stage == "critical":
        time = critical_time
    elif frame.stage == "depart":
        time = critical_time + frame.progress * (motion.time[-1] - critical_time)
    else:
        time = motion.time[-1]
    return (
        time,
        np.interp(time, motion.time, motion.theta),
        np.interp(time, motion.time, motion.omega),
        motion.foot,
    )


def stage_message(key, stage):
    if stage == "after":
        return "New stance foot; impact reduces speed."
    if stage == "before":
        return "Next spoke touches: just BEFORE impact."
    if stage == "critical":
        return {
            "A": "Over upright, with speed to spare.",
            "B": "Zero speed BEFORE upright: now it reverses.",
            "C": "Over upright, moving uphill.",
        }[key]
    if stage == "depart":
        return {
            "A": "Gravity accelerates it toward the next foot.",
            "B": "Gravity brings it back to its starting angle.",
            "C": "It reaches the previous spoke uphill.",
        }[key]
    return {
        "A": "Enough positive speed to cross the barrier.",
        "B": "It climbs, but cannot cross the barrier.",
        "C": "Enough negative speed for an uphill step.",
    }[key]


def create_animation(motions, params, fps=20):
    """Build the three wheel views and their shared one-impact return map."""
    length, gravity = params["length"], params["gravity"]
    count, gamma = int(params["num_spokes"]), params["slope_angle"]
    alpha = np.pi / count
    c = np.cos(2 * alpha)
    delta = 4 * gravity / length * np.sin(alpha) * np.sin(gamma)
    omega_1 = np.sqrt(2 * gravity / length * (1 - np.cos(gamma - alpha)))
    omega_2 = -np.sqrt(2 * gravity / length * (1 - np.cos(gamma + alpha)))
    frames = make_frames()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "text.color": INK,
            "axes.labelcolor": INK,
            "font.size": 10,
            "savefig.facecolor": BACKGROUND,
        }
    )
    # Figure size changes the exported image; wheel axis limits below control
    # the physical area shown. All three wheel panels use the same scale.
    fig = plt.figure(figsize=(14.4, 9), facecolor=BACKGROUND)
    fig.text(
        0.045,
        0.958,
        "RIMLESS WHEEL  /  Why the return map has three branches",
        fontsize=21,
        weight="bold",
    )
    fig.text(
        0.045,
        0.922,
        rf"{count} spokes    $\alpha={alpha:.3f}$ rad    $\gamma={gamma:.2f}$ rad"
        rf"    $g={gravity:g}$ m/s²    $\ell={length:g}$ m    $m={params['mass']:g}$ kg"
        "     |     Positive motion is downhill →",
        fontsize=10,
        color=MUTED,
    )
    panels = []
    for column, motion in enumerate(motions):
        left = 0.045 + column * 0.325
        color = COLORS[motion.key]
        fig.text(
            left,
            0.865,
            TITLES[motion.key],
            weight="bold",
            fontsize=14,
            bbox={"facecolor": color, "edgecolor": "none", "pad": 6, "alpha": 0.35},
        )
        bound = r"\gamma-\alpha" if motion.initial_speed > 0 else r"\gamma+\alpha"
        relation = {"A": r"w>\omega_1", "B": r"0<w<\omega_1", "C": r"w<\omega_2"}[
            motion.key
        ]
        fig.text(
            left,
            0.829,
            rf"Start: $\theta={bound}$,  $w={motion.initial_speed:+.1f}$ rad/s;  ${relation}$",
            fontsize=10,
        )
        state_text = fig.text(left, 0.796, "", fontsize=10, color=MUTED)

        ax = fig.add_axes((left, 0.492, 0.285, 0.286))
        ax.set(
            xlim=(-1.55 * length, 1.55 * length),
            ylim=(-0.28 * length, 2.10 * length),
            aspect="equal",
        )
        ax.axis("off")
        ground_x = np.array([-1.8, 1.8]) * length
        ground_y = -np.tan(gamma) * ground_x
        ax.fill_between(ground_x, -0.5 * length, ground_y, color="#e3e8ed")
        ax.plot(ground_x, ground_y, color=MUTED, lw=1.6)
        (vertical,) = ax.plot([0, 0], [0, 1.18 * length], ":", color="#a1abb6", lw=1)
        (trail,) = ax.plot([], [], color=color, alpha=0.55, lw=2)
        spokes = [ax.plot([], [], color="#adb6bf", lw=2)[0] for _ in range(count)]
        (hub,) = ax.plot([], [], "o", color=INK, ms=10, zorder=5)
        (foot_marker,) = ax.plot(
            [], [], "o", color=color, mec=INK, mew=0.7, ms=7, zorder=6
        )
        arrow = FancyArrowPatch(
            (0, 0),
            (0, 0),
            arrowstyle="-|>",
            mutation_scale=15,
            color=INK,
            lw=1.8,
            zorder=7,
        )
        ax.add_patch(arrow)
        message = fig.text(left, 0.46, "", fontsize=10, weight="bold")
        fig.text(left, 0.412, FORMULAS[motion.key], fontsize=18)
        impact_text = fig.text(left, 0.375, "", fontsize=10, color=MUTED)
        panels.append(
            (
                spokes,
                hub,
                foot_marker,
                arrow,
                trail,
                vertical,
                state_text,
                message,
                impact_text,
            )
        )

    # Plot post-impact velocity against the next post-impact velocity, never
    # against intermediate angular velocities during the animated stance.
    map_ax = fig.add_axes((0.085, 0.09, 0.44, 0.235))
    style_axis(map_ax)
    map_ax.set_title(
        "ONE-IMPACT RETURN MAP", loc="left", fontsize=11, weight="bold", pad=12
    )
    map_ax.set(
        xlim=(-3, 2),
        ylim=(-2.5, 2.1),
        xlabel=r"$w=\dot\theta_n^+$  [rad/s]",
        ylabel=r"$P(w)=\dot\theta_{n+1}^+$  [rad/s]",
    )
    map_ax.plot([-3, 2], [-3, 2], "--", color=INK, lw=1.2, alpha=0.65)
    map_ax.text(1.2, 1.78, "identity", fontsize=8, color=MUTED)
    for boundary, label in ((omega_2, r"$\omega_2$"), (omega_1, r"$\omega_1$")):
        map_ax.axvline(boundary, color=THRESHOLD_COLOR, ls="--", lw=1)
        map_ax.text(boundary - 0.07, 1.77, label, ha="right", color=THRESHOLD_COLOR)
    values = (
        ("A", np.linspace(omega_1 + 1e-8, 2, 200)),
        ("B", np.linspace(omega_2 + 1e-8, omega_1 - 1e-8, 200)),
        ("C", np.linspace(-3, omega_2 - 1e-8, 200)),
    )
    for key, w in values:
        if key == "A":
            next_w = c * np.sqrt(w * w + delta)
        elif key == "B":
            next_w = -c * w
        else:
            next_w = -c * np.sqrt(w * w - delta)
        map_ax.plot(w, next_w, color=COLORS[key], lw=3)
        # Open branch endpoints: a threshold trajectory takes infinite time
        # to approach upright, so its next impact is undefined.
        ends = [0, -1] if key == "B" else ([0] if key == "A" else [-1])
        for index in ends:
            map_ax.plot(
                w[index],
                next_w[index],
                "o",
                color=COLORS[key],
                mfc="white",
                ms=4,
                mew=1,
            )
    map_points = []
    for motion in motions:
        w, next_w = motion.initial_speed, motion.after_state[1]
        (point,) = map_ax.plot(
            w, next_w, "o", ms=8, mec=INK, mfc="white", mew=1.2, zorder=7
        )
        map_ax.annotate(
            motion.key,
            (w, next_w),
            xytext=(8, -14),
            textcoords="offset points",
            weight="bold",
            fontsize=10,
        )
        map_points.append(point)

    fig.text(0.595, 0.325, "READING THE MOTION", fontsize=11, weight="bold")
    fig.text(
        0.595,
        0.25,
        "Solid color: supporting leg and stance foot\n"
        "Dashed spoke: next contact   •   Arrow: hub motion\n"
        "Panels use independent physical clocks",
        fontsize=10,
        color=MUTED,
        linespacing=1.6,
    )
    fig.text(
        0.595,
        0.213,
        rf"$c=\cos(2\alpha)={c:.3f}$    $D=4(g/\ell)\sin\alpha\sin\gamma={delta:.3f}$",
        fontsize=11,
    )
    fig.text(
        0.595,
        0.177,
        rf"Thresholds: $\omega_2={omega_2:.3f}$,  $\omega_1={omega_1:.3f}$ rad/s",
        fontsize=10,
    )
    explanation = fig.text(0.595, 0.106, "", fontsize=11, linespacing=1.6)
    fig.text(
        0.045,
        0.026,
        "Slowed playback with pauses; panels align events, not physical time. "
        "Each example ends just after one impact. Colors identify branches, not long-term basins.",
        fontsize=9,
        color=MUTED,
    )

    def update(index):
        frame = frames[index]
        for motion, panel, point in zip(motions, panels, map_points):
            (
                spokes,
                hub,
                foot_marker,
                arrow,
                trail,
                vertical,
                state_text,
                message,
                impact_text,
            ) = panel
            time, theta, omega, foot = frame_state(motion, frame)
            center = foot + length * np.array([np.sin(theta), np.cos(theta)])
            angles = theta + 2 * alpha * np.arange(count)
            tips = center - length * np.column_stack((np.sin(angles), np.cos(angles)))
            incoming = (-motion.direction) % count
            color = COLORS[motion.key]
            for spoke_index, (line, tip) in enumerate(zip(spokes, tips)):
                line.set_data([center[0], tip[0]], [center[1], tip[1]])
                line.set_color(color if spoke_index == 0 else "#adb6bf")
                line.set_linewidth(4 if spoke_index == 0 else 2)
                line.set_linestyle(
                    "--" if spoke_index == incoming and frame.stage != "after" else "-"
                )
                if spoke_index == incoming and frame.stage != "after":
                    line.set_color(INK)
            hub.set_data([center[0]], [center[1]])
            foot_marker.set_data([foot[0]], [foot[1]])
            vertical.set_data([foot[0], foot[0]], [foot[1], foot[1] + 1.18 * length])
            arrow_start = center + np.array([0, 0.16 * length])
            tangent = np.sign(omega) * np.array([np.cos(theta), -np.sin(theta)])
            arrow.set_positions(arrow_start, arrow_start + 0.4 * length * tangent)
            arrow.set_visible(abs(omega) > 1e-8)
            past = motion.time <= time
            path = motion.foot + length * np.column_stack(
                (np.sin(motion.theta[past]), np.cos(motion.theta[past]))
            )
            trail.set_data(np.r_[path[:, 0], center[0]], np.r_[path[:, 1], center[1]])
            state_text.set_text(
                rf"$t={time:.3f}$ s     $\theta={theta:+.3f}$ rad     $\dot\theta={omega:+.3f}$ rad/s"
            )
            message.set_text(stage_message(motion.key, frame.stage))
            if frame.stage == "after":
                impact_text.set_text(
                    rf"Impact: ${motion.omega[-1]:+.3f}\ \to\ {motion.after_state[1]:+.3f}$ rad/s"
                )
                point.set_markerfacecolor(color)
            else:
                impact_text.set_text("Hollow map point: predicted post-impact speed")
                point.set_markerfacecolor("white")
        explanation.set_text(
            "Filled points: measured first-impact outputs.\n"
            "Impact reduces speed; B reversed earlier under gravity."
            if frame.stage == "after"
            else "A and C cross upright; B turns before reaching it.\n"
            "Positive starts at γ − α; negative starts at γ + α."
        )
        return []

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=False,
        repeat=True,
        cache_frame_data=False,
    )
    update(0)
    return fig, animation, frames, update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-show", action="store_true", help="Export without a window."
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Preview without writing files."
    )
    parser.add_argument(
        "--fps", type=int, default=20, help="Playback frames per second."
    )
    parser.add_argument("--dpi", type=int, default=100, help="Export resolution.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("rimless_wheel_return_map_cases.gif"),
    )
    args = parser.parse_args()
    if args.fps <= 0 or args.dpi <= 0:
        parser.error("--fps and --dpi must be positive.")
    if args.output.suffix.lower() != ".gif":
        parser.error("--output must end in .gif")
    if args.no_show:
        plt.switch_backend("Agg")

    # Edit the physical parameters in generate_params(); the example velocities
    # live in simulate_cases(). It checks that they still select A, B and C.
    params = generate_params()
    motions = simulate_cases(params, samples_per_case=241)
    fig, animation, frames, update = create_animation(motions, params, args.fps)
    for motion in motions:
        print(
            f"{motion.key}: {motion.initial_speed:+.3f} -> {motion.after_state[1]:+.6f} rad/s; "
            f"impact at t={motion.time[-1]:.6f} s",
            flush=True,
        )
    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)

        def progress(index, total):
            if index % 40 == 0 or index == total - 1:
                print(f"Rendering frame {index + 1}/{total}", flush=True)

        animation.save(
            args.output,
            writer=ReturnMapGifWriter(fps=args.fps),
            dpi=args.dpi,
            progress_callback=progress,
        )
        update(next(i for i, frame in enumerate(frames) if frame.stage == "critical"))
        fig.savefig(args.output.with_suffix(".png"), dpi=args.dpi)
        print(f"Saved {args.output}", flush=True)
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
