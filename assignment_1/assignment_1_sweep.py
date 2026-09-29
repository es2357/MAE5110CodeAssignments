import numpy as np
import matplotlib.pyplot as plt

gravity = 9.81
spoke_length = 1.0
grid_n = 150
iters = 60

# Sweep gamma
N_fixed = 6
alpha_fixed = np.pi / N_fixed

gamma_sweep = np.linspace(0.001, 0.9, 60)

roa_vs_gamma, floquet_vs_gamma = [], []
valid_vs_gamma, omega_roll_vs_gamma = [], []

for gamma in gamma_sweep:

    N = N_fixed
    alpha = np.pi / N
    E0 = gravity/spoke_length


    forward_guard = gamma + alpha
    backward_guard = gamma - alpha

    c = np.cos(2 * alpha)
    dE = 2 * (gravity/spoke_length) * (np.cos(backward_guard) - np.cos(forward_guard))
    w_min = np.sqrt(2 * (gravity/spoke_length) * (1 - np.cos(backward_guard)))
    omega1 = w_min

    if gamma <= alpha:
        omega1 = w_min
    else:
        omega1 = 0.0
    omega2 = -np.sqrt(2 * (gravity/spoke_length) * (1 - np.cos(alpha + gamma)))

    w_star = np.sqrt(c**2 * dE / (1 - c**2))
    valid_gait = w_star > omega1
    floquet = c**2

    theta = np.linspace(backward_guard, forward_guard, grid_n)
    w = np.linspace(-3, 3, grid_n)

    theta_cur, w_cur = np.meshgrid(theta, w)

    for i in range(iters):

        E = 0.5 * w_cur**2 + (gravity/spoke_length) * np.cos(theta_cur)
        forward = np.zeros_like(theta_cur, dtype=bool)

        forward[(theta_cur > 1e-12) & (w_cur >= 0)] = True

        forward[(theta_cur > 1e-12) & (w_cur < 0) & (E < E0)] = True

        forward[(theta_cur < -1e-12) & (w_cur > 0) & (E > E0)] = True

        forward[(np.abs(theta_cur) <= 1e-12) & (w_cur > 0)] = True

        target = np.where(forward, forward_guard, backward_guard)

        w_exit_sq = w_cur**2 + 2*(gravity/spoke_length)*(np.cos(theta_cur) - np.cos(target))
        w_exit = np.sqrt(np.maximum(w_exit_sq, 0))
        w_exit = np.where(forward, w_exit, -w_exit)

        theta_cur = np.where(forward, backward_guard, forward_guard)
        w_cur = c * w_exit

    roa = np.mean(np.abs(np.abs(w_cur) - w_star) < 1e-3)

    roa_vs_gamma.append(roa)
    floquet_vs_gamma.append(floquet)
    valid_vs_gamma.append(valid_gait)
    omega_roll_vs_gamma.append(w_star)

roa_vs_gamma = np.array(roa_vs_gamma)
floquet_vs_gamma = np.array(floquet_vs_gamma)
valid_vs_gamma = np.array(valid_vs_gamma)

# Sweep N

gamma_fixed = 0.2

N_sweep = np.arange(6, 13)

roa_vs_N, floquet_vs_N = [], []
valid_vs_N, omega_roll_vs_N, alpha_vs_N = [], [], []

for N in N_sweep:

    gamma = gamma_fixed
    alpha = np.pi / N
    E0 = gravity/spoke_length

    forward_guard = gamma + alpha
    backward_guard = gamma - alpha

    c = np.cos(2 * alpha)
    dE = 2 * (gravity/spoke_length) * (np.cos(backward_guard) - np.cos(forward_guard))

    w_min = np.sqrt(2 * (gravity/spoke_length) * (1 - np.cos(backward_guard)))
    omega1 = w_min if gamma <= alpha else 0.0
    omega2 = -np.sqrt(2 * (gravity/spoke_length) * (1 - np.cos(alpha + gamma)))

    w_star = np.sqrt(c**2 * dE / (1 - c**2))
    valid_gait = w_star > omega1
    floquet = c**2

    theta = np.linspace(backward_guard, forward_guard, grid_n)
    w = np.linspace(-3, 3, grid_n)

    theta_cur, w_cur = np.meshgrid(theta, w)

    for _ in range(iters):

        E = 0.5 * w_cur**2 + (gravity/spoke_length) * np.cos(theta_cur)

        forward = np.zeros_like(theta_cur, dtype=bool)

        forward[(theta_cur > 1e-12) & (w_cur >= 0)] = True

        forward[(theta_cur > 1e-12) & (w_cur < 0) & (E < E0)] = True

        forward[(theta_cur < -1e-12) & (w_cur > 0) & (E > E0)] = True

        forward[(np.abs(theta_cur) <= 1e-12) & (w_cur > 0)] = True

        target = np.where(forward, forward_guard, backward_guard)

        w_exit_sq = w_cur**2 + 2*(gravity/spoke_length)*(np.cos(theta_cur) - np.cos(target))
        w_exit = np.sqrt(np.maximum(w_exit_sq, 0))
        w_exit = np.where(forward, w_exit, -w_exit)

        theta_cur = np.where(forward, backward_guard, forward_guard)
        w_cur = c * w_exit

    roa = np.mean(np.abs(np.abs(w_cur) - w_star) < 1e-3)

    roa_vs_N.append(roa)
    floquet_vs_N.append(floquet)
    valid_vs_N.append(valid_gait)
    omega_roll_vs_N.append(w_star)
    alpha_vs_N.append(alpha)

roa_vs_N = np.array(roa_vs_N)
floquet_vs_N = np.array(floquet_vs_N)
valid_vs_N = np.array(valid_vs_N)



# Plots

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

ax[0,0].plot(gamma_sweep, roa_vs_gamma, "o-", ms=3)
ax[0,0].axvline(alpha_fixed, ls="--", color="gray")
ax[0,0].set(xlabel=r"$\gamma$ [rad]", ylabel="RoA fraction", title=f"RoA vs. slope (N={N_fixed})")
ax[0,1].plot(gamma_sweep, floquet_vs_gamma, "o-", ms=3)
ax[0,1].axvline(alpha_fixed, ls="--", color="gray")
ax[0,1].set(xlabel=r"$\gamma$ [rad]", ylabel="Floquet multiplier", title=f"Floquet vs. slope (N={N_fixed})", ylim=(0,1))
ax[1,0].plot(N_sweep, roa_vs_N, "o-")
ax[1,0].set(xlabel="Number of spokes N", ylabel="RoA fraction", title=rf"RoA vs. N ($\gamma={gamma_fixed}$)")
ax[1,1].plot(N_sweep, floquet_vs_N, "o-")
ax[1,1].set(xlabel="Number of spokes N", ylabel="Floquet multiplier", title=rf"Floquet vs. N ($\gamma={gamma_fixed}$)", ylim=(0,1))

plt.tight_layout()
plt.show()