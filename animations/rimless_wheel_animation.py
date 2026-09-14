from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp

# ============================================================
# Parameters
# ============================================================
g = 9.81
l = 1.0
N = 8
alpha = np.pi / N
gamma = 0.08

# Initial state: right after impact (t+)
theta0 = gamma - alpha         # negative
theta_dot0 = 1.5               # pick any reasonable positive speed

# Number of animation frames
n_flow_frames = 120
n_pause_start = 15
n_pause_preimpact = 18
n_pause_postimpact = 22

# ============================================================
# Continuous rimless-wheel dynamics
# ============================================================
def dynamics(t, state):
    theta, theta_dot = state
    theta_ddot = (g / l) * np.sin(theta)
    return [theta_dot, theta_ddot]

def impact_event(t, state):
    theta, theta_dot = state
    return theta - (gamma + alpha)

impact_event.terminal = True
impact_event.direction = 1

# ============================================================
# Geometry helpers
# ============================================================
def ground_y(x):
    """Ground line: downhill to the right."""
    return -np.tan(gamma) * x

def hub_position(stance_foot, theta):
    """
    Stance foot is fixed on ground.
    Theta measured from upward vertical, positive downhill.
    """
    return stance_foot + np.array([
        l * np.sin(theta),
        l * np.cos(theta)
    ])

def spoke_endpoints(hub, theta):
    """
    Endpoints of all spokes.
    k = 0 is the current stance spoke endpoint.
    """
    pts = []
    for k in range(N):
        ang = theta + 2 * k * alpha
        pt = hub - np.array([l * np.sin(ang), l * np.cos(ang)])
        pts.append(pt)
    return np.array(pts)

# ============================================================
# Simulate one stance phase until impact
# ============================================================
# First solve just to find exact impact time
sol_impact = solve_ivp(
    dynamics,
    (0.0, 5.0),
    [theta0, theta_dot0],
    events=impact_event,
    max_step=1e-3,
    rtol=1e-10,
    atol=1e-12
)

if len(sol_impact.t_events[0]) == 0:
    raise RuntimeError("Impact was not reached. Try a larger initial theta_dot0.")

t_impact = sol_impact.t_events[0][0]
theta_minus, theta_dot_minus = sol_impact.y_events[0][0]

# Re-simulate on a clean uniform grid for animation frames
t_eval = np.linspace(0.0, t_impact, n_flow_frames)
sol = solve_ivp(
    dynamics,
    (0.0, t_impact),
    [theta0, theta_dot0],
    t_eval=t_eval,
    rtol=1e-10,
    atol=1e-12
)

theta_flow = sol.y[0]
theta_dot_flow = sol.y[1]

# Reset at impact
theta_plus = theta_minus - 2 * alpha
theta_dot_plus = theta_dot_minus * np.cos(2 * alpha)

# ============================================================
# Foot positions
# ============================================================
# Start with first stance foot at the origin
foot0 = np.array([0.0, 0.0])

# Hub at impact just before the switch
hub_minus = hub_position(foot0, theta_minus)

# The next downhill spoke is k = N-1 (same as angle theta - 2alpha)
endpoints_minus = spoke_endpoints(hub_minus, theta_minus)
foot1 = endpoints_minus[N - 1]   # this becomes the new stance foot after impact

# ============================================================
# Build frame list
# ============================================================
frames = []

# Hold at the initial post-impact configuration for a moment
for _ in range(n_pause_start):
    frames.append({
        "mode": "start_pause",
        "theta": theta0,
        "theta_dot": theta_dot0,
        "foot": foot0
    })

# Continuous motion from theta = gamma - alpha up to impact
for th, thd in zip(theta_flow, theta_dot_flow):
    frames.append({
        "mode": "flow",
        "theta": th,
        "theta_dot": thd,
        "foot": foot0
    })

# Pause just before impact
for _ in range(n_pause_preimpact):
    frames.append({
        "mode": "preimpact",
        "theta": theta_minus,
        "theta_dot": theta_dot_minus,
        "foot": foot0
    })

# Pause just after impact / reset
for _ in range(n_pause_postimpact):
    frames.append({
        "mode": "postimpact",
        "theta": theta_plus,
        "theta_dot": theta_dot_plus,
        "foot": foot1
    })

# ============================================================
# Plot setup
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_aspect("equal")

# Plot limits
x_min = -1.4
x_max = foot1[0] + 1.2
xs = np.linspace(x_min, x_max, 400)
ys = ground_y(xs)

ax.set_xlim(x_min, x_max)
ax.set_ylim(np.min(ys) - 0.35, 1.5)
ax.set_title("Rimless Wheel: one spoke from t+ to handoff")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Static ground line
ground_line, = ax.plot(xs, ys, color="black", linewidth=2)

# Artists to update
vertical_ref, = ax.plot([], [], "k--", linewidth=1.5, alpha=0.7)
hub_dot, = ax.plot([], [], "ko", markersize=8)
stance_foot_dot, = ax.plot([], [], "o", color="crimson", markersize=8)
old_foot_dot, = ax.plot([], [], "o", color="lightcoral", markersize=6, alpha=0.6)

# One line per spoke
spoke_lines = []
for _ in range(N):
    line, = ax.plot([], [], color="0.75", linewidth=2)
    spoke_lines.append(line)

status_text = ax.text(
    0.02, 0.98, "", transform=ax.transAxes,
    va="top", ha="left", fontsize=11,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7")
)

# ============================================================
# Animation update
# ============================================================
def update(frame_idx):
    frame = frames[frame_idx]
    mode = frame["mode"]
    theta = frame["theta"]
    theta_dot = frame["theta_dot"]
    foot = frame["foot"]

    hub = hub_position(foot, theta)
    endpoints = spoke_endpoints(hub, theta)

    # Draw all spokes in gray first
    for k, line in enumerate(spoke_lines):
        xdata = [hub[0], endpoints[k, 0]]
        ydata = [hub[1], endpoints[k, 1]]
        line.set_data(xdata, ydata)
        line.set_color("0.75")
        line.set_linewidth(2)
        line.set_linestyle("-")
        line.set_alpha(0.9)

    # Highlight current stance spoke (k=0)
    spoke_lines[0].set_color("royalblue")
    spoke_lines[0].set_linewidth(4)

    # Highlight next downhill spoke (k=N-1)
    spoke_lines[N - 1].set_color("darkorange")
    spoke_lines[N - 1].set_linewidth(3)
    spoke_lines[N - 1].set_linestyle("--")

    # Vertical reference through the stance foot
    vertical_ref.set_data(
        [foot[0], foot[0]],
        [foot[1], foot[1] + 1.25 * l]
    )

    # Hub and stance foot
    hub_dot.set_data([hub[0]], [hub[1]])
    stance_foot_dot.set_data([foot[0]], [foot[1]])

    # Show previous foot faintly after the handoff
    if mode == "postimpact":
        old_foot_dot.set_data([foot0[0]], [foot0[1]])
    else:
        old_foot_dot.set_data([], [])

    # Status text
    if mode == "start_pause":
        label = r"Start at $t^+$: new stance spoke, $\theta=\gamma-\alpha<0$"
    elif mode == "flow":
        if theta < -1e-2:
            label = r"$\theta<0$: stance spoke leans uphill"
        elif abs(theta) <= 1e-2:
            label = r"$\theta\approx 0$: stance spoke is vertical"
        else:
            label = r"$\theta>0$: moving toward the next impact"
    elif mode == "preimpact":
        label = r"$t^-$: just before impact, $\theta=\gamma+\alpha$"
    elif mode == "postimpact":
        label = (
            r"$t^+$ after reset: handoff to next spoke" "\n"
            r"$\theta^+ = \theta^- - 2\alpha = \gamma-\alpha$"
        )
    else:
        label = ""

    status_text.set_text(
        label +
        f"\nθ = {theta:+.3f} rad"
        f"\nθ̇ = {theta_dot:+.3f} rad/s"
    )

    return (
        [ground_line, vertical_ref, hub_dot, stance_foot_dot, old_foot_dot, status_text]
        + spoke_lines
    )

# ============================================================
# Make animation
# ============================================================
anim = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=60,
    blit=False,
    repeat=True
)

plt.tight_layout()

# Show the animation live
plt.show()

# ------------------------------------------------------------
# Optional: save as GIF
# ------------------------------------------------------------

writer = PillowWriter(fps=20)
output = Path(__file__).with_name("rimless_wheel_one_step.gif")
anim.save(output, writer=writer)
print(f"Saved to {output}")
