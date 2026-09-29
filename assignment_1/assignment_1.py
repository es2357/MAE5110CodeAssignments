import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from models import rimlessWheel as model

gravity = 9.81
mass = 1.0
spoke_length = 1.0
num_spokes = 6
gamma = 0.2     #Absolute global vertical
alpha = np.pi / num_spokes      #Half-spoke angle

params = {
    "gravity": gravity,
    "mass": mass,
    "spoke_length": spoke_length,
    "gamma": gamma,
    "num_spokes": num_spokes,
    "alpha": alpha,
}

initial_state = np.array([gamma-alpha, 1.3])
forward_guard = gamma + alpha
backward_guard = gamma - alpha

sim_time = 10.0
max_steps = 200
time_steps = 1e-3

t0 = 0.0
state = np.array(initial_state,dtype=float)

t_history, state_history = [], []
t_impact, pre_impact_history, post_impact_history = [], [], []
walking = False

for step in range(max_steps):
    sol = solve_ivp(
        model.dynamics,
        [t0, sim_time],
        state,
        args=(params,),
        events = [model.detect_forward_event, model.detect_backward_event],
        max_step = time_steps, rtol=1e-10, atol = 1e-12, dense_output = True,
    )
    
    t_history.append(sol.t)
    state_history.append(sol.y)
    
    if sol.t_events[0].size > 0:
        t_pre = sol.t_events[0][0]
        state_pre_impact = sol.y_events[0][0]
        state_post_impact = model.reset_map(state_pre_impact, params)
        
        t_impact.append(t_pre)
        pre_impact_history.append(state_pre_impact)
        post_impact_history.append(state_post_impact)
        
        state = state_post_impact
        t0 = t_pre
        if t0 >= sim_time:
            break
        
    elif sol.t_events[1].size > 0:
        t_imp = sol.t_events[1][0]
        state_pre_impact = sol.y_events[1][0]
        theta_pre, w_pre = state_pre_impact
        state_post_impact = np.array([theta_pre + 2*alpha, w_pre * np.cos(2*alpha)])

        t_impact.append(t_imp)
        pre_impact_history.append(state_pre_impact)
        post_impact_history.append(state_post_impact)

        state = state_post_impact
        t0 = t_imp
        if t0 >= sim_time:
            break
    else:
        break

t = np.concatenate(t_history)
state = np.concatenate(state_history, axis = 1)

energy = np.zeros(len(t))
PE = np.zeros(len(t))
KE = np.zeros(len(t))

for i in range(len(t)):
    potential_energy, kinetic_energy = model.energy_calc(state[:, i], params)

    PE[i] = potential_energy
    KE[i] = kinetic_energy
    energy[i] = potential_energy + kinetic_energy

# Plot energy versus time
plt.figure()
plt.plot(t, PE, label = "Potential Energy", color = 'blue')
plt.plot(t,KE, label = "Kinetic Energy", color = "red")
plt.plot(t, energy, label = "Total Energy", color = "green")
plt.xlabel("Time [s]")
plt.ylabel("Total Energy [J]")
plt.title("Rimless Wheel Energy vs. Time")
plt.legend(loc='upper right', fontsize=10, frameon=True)
plt.grid(True)
plt.show()

# Check kinetic energy
for pre, post in zip(pre_impact_history, post_impact_history):
    ke_pre  = 0.5 * mass * spoke_length**2 * pre[1]**2
    ke_post = 0.5 * mass * spoke_length**2 * post[1]**2
    print(f"KE_pre={ke_pre:.4f}  KE_post={ke_post:.4f}  "
          f"ratio={ke_post/ke_pre:.4f}  expected={np.cos(2*alpha)**2:.4f}")