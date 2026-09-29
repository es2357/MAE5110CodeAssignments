import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# Paramaters
gravity = 9.81
mass = 1.0
spoke_length = 1.0

num_spokes = 6
gamma = 0.2

alpha = np.pi / num_spokes

forward_guard = gamma + alpha
backward_guard = gamma - alpha

# Numerical tolerance for checking theta = 0
tol = 1e-12


# Physics
delta_energy = (2 * (gravity / spoke_length) * (np.cos(backward_guard) - np.cos(forward_guard)))
delta_energy_joules = (0.5 * mass * spoke_length**2 * delta_energy)

reset_vel_scale = np.cos(2 * alpha)
omega_roll_star = np.sqrt(reset_vel_scale**2 * delta_energy/ (1 - reset_vel_scale**2))


# Forward velocity threshold
angular_vel_min = np.sqrt(2* (gravity / spoke_length)* (1 - np.cos(backward_guard)))

if gamma <= alpha:
    omega_forward_threshold = angular_vel_min
else:
    omega_forward_threshold = 0.0
    

# Backward velocity threshold
angular_vel_min_backward = np.sqrt(2* (gravity / spoke_length)* (1 - np.cos(alpha + gamma)))
omega_backward_threshold = -angular_vel_min_backward


# Print variables for debugging
print(f"N={num_spokes}, gamma={gamma} rad, "f"alpha={alpha:.4f} rad " f"(gamma<=alpha: {gamma <= alpha})")
print("delta_energy " "(specific change in omega^2):",delta_energy,)
print("delta_energy (actual Joules):", delta_energy_joules,)
print("omega_roll_star:",omega_roll_star,)
print("omega_forward_threshold:", omega_forward_threshold, "(raw formula:", angular_vel_min,")",)
print("omega_backward_threshold:", omega_backward_threshold,)
print("valid gait: omega_roll_star > " "omega_forward_threshold ->",omega_roll_star > omega_forward_threshold,)


# RoA
theta_grid = np.linspace(backward_guard, forward_guard, 300,)
angular_vel_grid = np.linspace(-3, 3, 300,)

energy_barrier = gravity / spoke_length
theta_cur, w_cur = np.meshgrid(theta_grid, angular_vel_grid,)

theta_cur = theta_cur.copy()
w_cur = w_cur.copy()

iterations = 100


for i in range(iterations):

    energy = (0.5 * w_cur**2 + (gravity / spoke_length) * np.cos(theta_cur))

    reaches_forward = np.zeros_like(theta_cur, dtype=bool,)

    mask_pos = theta_cur > tol
    reaches_forward[mask_pos & (w_cur >= 0)] = True
    reaches_forward[mask_pos & (w_cur < 0) & (energy < energy_barrier)] = True

    mask_neg = theta_cur < -tol
    reaches_forward[mask_neg & (w_cur > 0) & (energy > energy_barrier)] = True

    mask_zero = np.abs(theta_cur) <= tol
    reaches_forward[mask_zero & (w_cur > 0)] = True

    target_angle = np.where(reaches_forward, forward_guard, backward_guard,)


    w_exit_sq = (w_cur**2 + 2 * (gravity / spoke_length)* (np.cos(theta_cur) - np.cos(target_angle)))
    w_exit_mag = np.sqrt(np.maximum(w_exit_sq, 0.0))
    w_exit = np.where(reaches_forward, w_exit_mag, -w_exit_mag,)
    
    theta_cur = np.where(reaches_forward, backward_guard, forward_guard,)

    w_cur = (w_exit * reset_vel_scale)



convergence_tol = 1e-3

classes = np.where(np.abs(w_cur - omega_roll_star) < convergence_tol, 1, 0,)

print("fraction converging to forward rolling:", (classes == 1).mean(),)

# Plot RoA
theta_cycle = np.linspace(backward_guard, forward_guard, 500,)
angular_vel_cycle = np.sqrt(omega_roll_star**2 + 2 * (gravity / spoke_length)* (np.cos(backward_guard) - np.cos(theta_cycle)))

cmap = ListedColormap([ "#3b0f70", "#f9c932",])
norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N,)

fig, ax = plt.subplots(figsize=(9, 6.5))
mesh = ax.pcolormesh(theta_grid, angular_vel_grid, classes, cmap=cmap, norm=norm, shading="auto",)

ax.plot(theta_cycle, angular_vel_cycle, linewidth=2, color="tab:blue", label="Stable rolling cycle",)
ax.scatter([backward_guard], [omega_roll_star], s=60, color="tab:blue", zorder=6, label=(rf"$\omega^*_{{roll}}" rf"={omega_roll_star:.4f}$"),)
ax.set_xlabel(r"$\theta$ [rad]")
ax.set_ylabel(r"$\dot{\theta}$ [rad/s]")
ax.set_title("Region of Attraction\n"f"(N={num_spokes}, "f"$\\gamma$={gamma} rad)")
ax.legend(loc="lower right", fontsize=8,)

cbar = fig.colorbar(mesh, ax=ax, ticks=[0, 1],)
cbar.ax.set_yticklabels(["Does not converge to forward rolling", "Converges to forward rolling",])

plt.show()

# Poincare Return Map

def full_return_map(w):

    w = np.asarray(w,dtype=float,)

    out = np.full_like(w,np.nan,)

    branch1 = (w > omega_forward_threshold)
    out[branch1] = (reset_vel_scale* np.sqrt(w[branch1]**2 + delta_energy))

    branch2 = ((omega_backward_threshold < w) & (w <= omega_forward_threshold))
    out[branch2] = (-w[branch2]* reset_vel_scale)

    branch3 = ( w <= omega_backward_threshold)

    out[branch3] = (-reset_vel_scale* np.sqrt(np.maximum(w[branch3]**2 - delta_energy, 0.0,)))

    return out


# Plot return map

w_n = np.linspace(-8.0, 6.0, 2000,)
w_np1 = full_return_map(w_n)

fig, ax = plt.subplots(figsize=(10, 6.5))

ax.plot(w_n, w_np1, color="blue", lw=2,label="Return map",)
ax.plot(w_n, w_n, color="black", ls="--",lw=1.5, label="Identity line",)
ax.axvline(omega_forward_threshold, color="green", ls="-.", lw=2,)
ax.axvline(omega_backward_threshold, color="green", ls="-.", lw=2, label=r"$\omega_2$ and $\omega_1$ thresholds",)
ax.annotate(r"$\omega_1$",(omega_forward_threshold, 5.5,), textcoords="offset points", xytext=(5, 0),)
ax.annotate(r"$\omega_2$",(omega_backward_threshold, 5.5,), textcoords="offset points", xytext=(-25, 0),)


if gamma < alpha:
    ax.plot(0, 0, ".", color="red", ms=10, zorder=6, label=r"Zero-velocity map fixed point $\omega^*=0$",)
    ax.annotate(r"$\omega^*=0$", (0, 0), textcoords="offset points", xytext=(-10, -25),color="red",)


ax.plot(omega_roll_star, omega_roll_star, ".", color="red", ms=10, zorder=6, label=r"Rolling fixed point $\omega^*_{roll}$",)
ax.annotate(rf"$\omega^*_{{roll}}={omega_roll_star:.3f}$", (omega_roll_star, omega_roll_star,), textcoords="offset points", xytext=(10, -25), color="red",)

ax.set_xlabel(r"$\dot{\theta}_n^+$ "r"[angular velocity after collision $n$]")
ax.set_ylabel(r"$\dot{\theta}_{n+1}^+$ " r"[angular velocity after collision $n+1$]")

ax.set_title(f"Return Map "f"(N={num_spokes}, "f"$\\gamma$={gamma} rad)")

ax.set_xlim(-8, 6)
ax.set_ylim(-8, 6)

ax.grid(True,ls=":",alpha=0.5,)
ax.legend(loc="lower right", fontsize=9,)

plt.show()


# Floquet Multiplier
eps = 1e-3

w_minus = (omega_roll_star - eps)
w_plus = (omega_roll_star + eps)

f_minus = full_return_map(w_minus)
f_plus = full_return_map(w_plus)

floquet_estimate = (f_plus - f_minus) / (2 * eps)
floquet_analytic = (reset_vel_scale**2)

print()

print("----- Floquet Multiplier -----")
print(f"omega_roll_star = "f"{omega_roll_star:.6f}")
print(f"perturbation eps = "f"{eps}")
print(f"w- = {w_minus:.6f} "f"-> P(w-) = {float(f_minus):.6f}")
print(f"w+ = {w_plus:.6f} "f"-> P(w+) = {float(f_plus):.6f}")
print("Floquet multiplier ""(finite-difference estimate) = "f"{float(floquet_estimate):.6f}")
print("Floquet multiplier ""(analytic cos^2(2*alpha)) = "f"{floquet_analytic:.6f}")
print("difference = "f"{abs(float(floquet_estimate) - floquet_analytic):.2e}")

print("|multiplier| < 1 -> " "rolling limit cycle is locally stable:", abs(float(floquet_estimate)) < 1,)