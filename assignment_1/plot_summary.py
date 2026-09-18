"""
SUMMARY PLOTS
: compares energy loss, rolling RoA, and local convergence;


The impact multiplies angular velocity by cos(2*pi/N);
fraction lost is 1 - cos(2*pi/N)**2.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    import summary_sim
else:
    from . import summary_sim

DATA_DIR = Path(__file__).resolve().parent / "assignment_1_data"
SPOKE_COUNTS = summary_sim.SPOKE_COUNTS
GAMMA_RANGE = (0.0, 0.4)  # radians
COMPARISON_INCLINES = (0.08, 0.16, 0.26)


def energy_loss_percent(spokes):
    """Percentage of the pre-impact kinetic energy lost in one collision."""
    alpha = np.pi / spokes
    return 100 * (1 - np.cos(2 * alpha) ** 2)


def plot_energy_loss():
    """Plot loss against incline, with one horizontal line per spoke count."""
    figure, axis = plt.subplots(figsize=(9, 5.5), layout="constrained")

    for spokes in SPOKE_COUNTS:
        loss = energy_loss_percent(spokes)
        axis.plot(
            GAMMA_RANGE,
            [loss, loss],
            linewidth=2.5,
            label=f"N = {spokes}: {loss:.1f}%",
        )

    axis.set(
        xlabel=r"Incline $\gamma$ (rad)",
        ylabel="Kinetic energy lost per impact (%)",
        title=(
            "Rimless wheel: energy loss per impact\n"
            r"$100[1-\cos^2(2\pi/N)]$% of pre-impact kinetic energy"
        ),
        xlim=GAMMA_RANGE,
        ylim=(0, 100),
        yticks=np.arange(0, 101, 10),
    )
    axis.grid(alpha=0.25)
    axis.legend(title="Spokes and loss", loc="center left", bbox_to_anchor=(1.02, 0.5))
    return figure


def case_values(rows, spokes=None, gamma=None):
    """Select one curve, ordered by its changing parameter."""
    selected = [
        row for row in rows
        if (spokes is None or row["num_spokes"] == spokes)
        and (gamma is None or np.isclose(row["gamma"], gamma))
    ]
    variable = "gamma" if spokes is not None else "num_spokes"
    return sorted(selected, key=lambda row: row[variable])


def plot_roa_curve(axis, cases, variable, color, label, marker="o"):
    """Unknown outcomes extend the possible rolling fraction upward."""
    axis.errorbar(
        [row[variable] for row in cases],
        [row["rolling_percent"] for row in cases],
        yerr=[np.zeros(len(cases)), [row["unresolved_percent"] for row in cases]],
        color=color, marker=marker, markersize=5, linewidth=1.6,
        capsize=3, label=label,
    )


def format_roa_axis(axis):
    axis.set(
        title="How much of the sampled state space reaches rolling?",
        ylabel="Initial states classified as rolling (%)",
        ylim=(-3, 103), yticks=np.arange(0, 101, 20),
    )


def format_multiplier_axis(axis):
    axis.axhline(1, color="gray", linestyle="--", linewidth=1)
    axis.text(0.02, 0.96, r"$|\lambda| = 1$: stability boundary",
              transform=axis.transAxes, fontsize=9, va="top")
    axis.set(
        title="How much speed error remains after one impact?",
        ylabel=r"Floquet multiplier $\lambda$ (smaller = faster decay)",
        ylim=(0, 1.08),
    )


def finish_comparison(figure, axes, title, note):
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(title, fontsize=15)
    figure.text(0.5, 0.025, note, ha="center", va="bottom", fontsize=9)
    figure.tight_layout(rect=(0, 0.13, 1, 0.95))
    return figure


def plot_incline_summary(rows):
    """Compare basin size and per-impact convergence as the slope changes."""
    figure, (roa_axis, multiplier_axis) = plt.subplots(1, 2, figsize=(12, 5.7))
    for index, spokes in enumerate(SPOKE_COUNTS):
        cases = case_values(rows, spokes=spokes)
        color = f"C{index}"
        plot_roa_curve(roa_axis, cases, "gamma", color, f"N = {spokes}")
        inclines = [row["gamma"] for row in cases]
        multiplier_axis.plot(
            inclines, [row["exact_multiplier"] for row in cases],
            color=color, linewidth=1.6,
        )
        multiplier_axis.plot(
            inclines, [row["multiplier"] for row in cases],
            "o", color=color, markersize=5,
        )

    format_roa_axis(roa_axis)
    format_multiplier_axis(multiplier_axis)
    roa_axis.legend(ncol=2, fontsize=9)
    for axis in (roa_axis, multiplier_axis):
        axis.set(xlabel=r"Incline $\gamma$ (rad)", xlim=(0.03, 0.40))
    return finish_comparison(
        figure, (roa_axis, multiplier_axis), "Effect of incline on rolling and local convergence",
        "RoA: 21 × 41 equally spaced starting states; lines connect sampled cases.\n"
        "Floquet: lines are analytic, dots are simulation estimates; no multiplier without a rolling cycle.\n"
        r"Only $\gamma < \pi/N$ is included. RoA error bars, if present, show unresolved outcomes.",
    )


def plot_spoke_summary(rows):
    """Compare N at three inclines that are supported for every spoke count."""
    figure, (roa_axis, multiplier_axis) = plt.subplots(1, 2, figsize=(12, 5.7))
    markers = ("o", "s", "^")
    multiplier_axis.plot(
        list(SPOKE_COUNTS), [np.cos(2 * np.pi / n) ** 2 for n in SPOKE_COUNTS],
        color="black", linewidth=1.5, label=r"Analytic $\cos^2(2\pi/N)$",
    )
    for index, gamma in enumerate(COMPARISON_INCLINES):
        cases = case_values(rows, gamma=gamma)
        label = rf"$\gamma = {gamma:.2f}$ rad"
        color = f"C{index}"
        plot_roa_curve(roa_axis, cases, "num_spokes", color, label, markers[index])
        # Nested, open markers make coincident estimates visible.
        multiplier_axis.plot(
            [row["num_spokes"] for row in cases],
            [row["multiplier"] for row in cases],
            linestyle="none", marker=markers[index], color=color,
            markersize=11 - 3 * index, markerfacecolor="none", label=label,
        )

    format_roa_axis(roa_axis)
    format_multiplier_axis(multiplier_axis)
    roa_axis.legend(fontsize=9)
    multiplier_axis.legend(loc="lower right", fontsize=9)
    for axis in (roa_axis, multiplier_axis):
        axis.set(xlabel="Number of spokes N", xticks=list(SPOKE_COUNTS), xlim=(5.8, 12.2))
    return finish_comparison(
        figure, (roa_axis, multiplier_axis), "Effect of spoke count on rolling and local convergence",
        r"RoA uses the same 21 × 41 grid in $(\theta-\gamma)/(\pi/N)\in[-1,1]$ and $\omega\in[-3,3]$ rad/s."
        "\nThe angle interval changes with N; percentages describe the sampled box, not absolute basin area.\n"
        "Floquet estimates overlap across inclines wherever a rolling cycle exists.",
    )


def decay_rate(row):
    """Local error envelope: |error(t)| is proportional to exp(-rate*t)."""
    multiplier = row["multiplier"]
    period = row["step_time"]
    if np.isfinite(multiplier) and 0 < abs(multiplier) < 1 and period > 0:
        return -np.log(abs(multiplier)) / period
    return np.nan


def plot_convergence_time(rows):
    """Include step duration when comparing local convergence per second."""
    figure, (incline_axis, spoke_axis) = plt.subplots(1, 2, figsize=(12, 5.7))
    for index, spokes in enumerate(SPOKE_COUNTS):
        cases = case_values(rows, spokes=spokes)
        incline_axis.plot(
            [row["gamma"] for row in cases], [decay_rate(row) for row in cases],
            "o-", color=f"C{index}", markersize=5, label=f"N = {spokes}",
        )
    for index, gamma in enumerate(COMPARISON_INCLINES):
        cases = case_values(rows, gamma=gamma)
        spoke_axis.plot(
            [row["num_spokes"] for row in cases], [decay_rate(row) for row in cases],
            "o-", color=f"C{index}", markersize=5, label=rf"$\gamma = {gamma:.2f}$ rad",
        )

    incline_axis.set(title="Changing incline", xlabel=r"Incline $\gamma$ (rad)")
    spoke_axis.set(title="Changing spoke count", xlabel="Number of spokes N", xticks=list(SPOKE_COUNTS))
    for axis in (incline_axis, spoke_axis):
        axis.set(ylabel=r"Local decay rate $-\ln|\lambda|/T$ (s$^{-1}$)", ylim=(0, None))
        axis.legend(fontsize=9, ncol=2 if axis is incline_axis else 1)
    return finish_comparison(
        figure, (incline_axis, spoke_axis), "Local convergence per second: higher means faster decay",
        r"Near steady rolling, $|e_k|\approx|\lambda|^k|e_0|$ and $t\approx kT$, where T is the steady step time."
        "\nThis is the local decay of small post-impact speed errors, not the time to settle from an arbitrary state.\n"
        "Dots use simulated multipliers and step times; gaps mean no rolling cycle or no resolved estimate.",
    )


def save_plot(figure, filename):
    output_path = DATA_DIR / filename
    figure.savefig(output_path, dpi=250)
    print(f"Saved plot to: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save the figures without opening plot windows.",
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help="Number of parallel workers for missing RoA cases (default: 3).",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    figures = [plot_energy_loss()]
    save_plot(figures[0], "rimless_wheel_energy_loss_summary.png")
    rows = summary_sim.build_summary(workers=args.workers)
    for plot, filename in (
        (plot_incline_summary, "rimless_wheel_incline_summary.png"),
        (plot_spoke_summary, "rimless_wheel_spoke_summary.png"),
        (plot_convergence_time, "rimless_wheel_convergence_time_summary.png"),
    ):
        figure = plot(rows)
        save_plot(figure, filename)
        figures.append(figure)

    if not args.no_show:
        plt.show()
    for figure in figures:
        plt.close(figure)


if __name__ == "__main__":
    main()
