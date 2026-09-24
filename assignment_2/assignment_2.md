

# Assignment 2

Ekaterina Skorniakova (es2357)

The goal of Assignment 2 was to develop continuous ankle-torque and discrete foot-placement controllers that bring an inverted pendulum walker to a stable standing equilibrium in as few steps as possible, using feedback linearization, Poincaré maps, and grid-based lookup tables.

## Setup and Dynamics

The setup of the inverted pendulum walker (IPW) is similar to that of the rimless-wheel. The angle $\gamma = 0.06$ is the incline of the plane. The angle $\alpha$ here, however, is half the angle between the stance and swing legs and can be chosen once per stance phase. This allows the walker to adjust its foot placement at each step.

The walker has two control inputs: the ankle torque $\tau$, applied continuously during stance, and the foot-placement angle $\alpha$, selected discretely for each step. These inputs are constrained by
$$
\alpha \in \left[\frac{\pi}{8}, \frac{\pi}{7}\right],
\qquad
\tau \in [-0.1mg\ell,\;0.05mg\ell].
$$

When the swing leg contacts the ground, an instantaneous impact changes the angular velocity and the swing leg becomes the new stance leg.

<img src="IPW_setup.png" alt="" width="700">

The objective is to choose foot placements that bring the walker into the region of attraction of an upright balancing controller in as few steps as possible. Outside this region, the ankle torque is zero; once the walker enters it, the ankle controller stabilizes the standing equilibrium at $(\theta,\dot{\theta})=(0,0)$.

There are three cases for the IPW illustrated in the sketches: midstance, impact, and backward falling.

<img src="IPW_cases.png" alt="" width="700">

At midstance, the stance leg is vertically upright, so $\theta = 0$, and the walker is moving forward with $\dot{\theta} > 0$.

At impact, the swing foot contacts the ground when $\theta^- = \gamma + \alpha$. The swing leg becomes the new stance leg, and the state changes according to

$$
\theta^+ = \gamma - \alpha,
\qquad
\dot{\theta}^+ = \dot{\theta}^-\cos(2\alpha),
$$

where $-$ and $+$ indicate the states immediately before and after impact. Choosing a larger $\alpha$ increases the step length and reduces the fraction of angular velocity retained after impact. It also changes the touchdown angle, shifting the impact guard in state space.

The failure case occurs when the walker has insufficient forward motion to reach the upright position and is outside the balancing controller's region of attraction. It first stops at $\theta < 0$ with $\dot{\theta} = 0$, then begins falling backward with $\dot{\theta} < 0$. The foot-placement policy must avoid this outcome while guiding the walker toward the region where the ankle controller can bring it to rest.

The angle $\alpha$ sweeps between $\left[\frac{\pi}{8}, \frac{\pi}{7}\right]$, producing the event guards as shown below.


<img src="IPW_phase_plot.png" alt="" width="700">



Adding the ankle torque $\tau$ changes the vertical component of the vector field in the $(\theta,\dot{\theta})$ phase plot, since $\ddot{\theta} = (g/\ell)\sin\theta + \tau/(m\ell^2)$. At any state, increasing $\tau$ shifts the vector upward by increasing angular acceleration, and decreasing $\tau$ shifts it downward.
 For forward motion, where $\dot{\theta} > 0$, positive torque speeds up the walker relative to the passive dynamics, and negative torque slows it down.
With constant ankle torque, equilibrium requires $\dot{\theta}=0$ and $\sin\theta_e=-\tau/(mg\ell)$. The equilibrium near upright is shifted from vertical but remains unstable, and stabilizing it requires feedback control.


## Visualization of RoA

The ankle controller combines gravity cancellation with proportional and derivative feedback, using $k_p = 4$ and $k_d = 4$ (giving a critically damped response) and respecting the torque limits. Its region of attraction (RoA) was first estimated on a $9 \times 13$ grid of initial states, with $\theta_0 \in [-0.1, 0.1]$ rad and $\dot{\theta}_0 \in [-0.4, 0.4]$ rad/s. Each trial ran for up to 5 s. An initial state was classified as belonging to the estimated RoA if its trajectory satisfied $|\theta| < 10^{-3}$ rad and $|\dot{\theta}| < 10^{-3}$ rad/s throughout the final 0.5 s of the 5 s simulation, with no impact or departure from the allowed angle range during the simulation.

<img src="../output/assignment_2/roa.png" alt="" width="700">

The green region shows the initial states that the controller successfully brought to upright. Motion toward vertical helps the walker recover from an initial lean.
The controller can apply twice as much negative torque as positive torque (from torque limits), allowing it to resist forward motion more strongly than backward motion. This may be why the green region is asymmetric.


### RoA Grid Resolution

To examine the effect of grid resolution, I sampled the same initial-state range using grids of $9\times13$, $17\times25$, and $33\times49$ points (coarser to finer in the images). The controller parameters, simulation duration, and convergence criteria were kept fixed.

<img src="9_times_13.png" alt="" width="700">
<img src="17_times_25.png" alt="" width="700">
<img src="33_times_49.png" alt="" width="700">


All three grids have a similar shape as before. The finer grids provide more detail near the transition between initial states that converge and those for which convergence is not confirmed. Grid resolution also affects the controller’s RoA-entry test. Since the implementation classifies a state between samples as inside the RoA only when all surrounding samples have converged, a coarse grid can exclude recoverable states near the boundary, changing when the walker switches from walking to balancing.




## Poincaré Section

I chose midstance as the Poincaré section because it occurs at a fixed angle, $\theta = 0$, with $\dot{\theta} > 0$, regardless of the chosen foot-placement angle $\alpha$, and the section would be transverse to the flow. The touchdown angle, $\theta = \gamma + \alpha$, changes with the control input, making it less convenient. Fixing $\theta$ at midstance allows the step-to-step state to be described by angular velocity alone, simplifying the return map and reducing the grid to one state dimension and one control dimension.


The return map relates the angular velocity at one midstance crossing to the next:

$$
\dot{\theta}_{k+1} = P(\dot{\theta}_k,\alpha_k),
$$

where $\alpha_k$ is the foot-placement angle selected for that step. The next state is recorded when the walker returns to $\theta = 0$ with positive angular velocity after impact.

<img src="../output/assignment_2/poincare_return_map.png" alt="" width="600">

The return map above uses the initial 11-velocity, 5-angle grid and also shows why midstance is a convenient Poincaré section.
Each colored curve shows how a starting midstance velocity maps to the next midstance velocity for a particular $\alpha$. The dashed line represents unchanged speed, and points below it indicate slowing between crossings. Larger angles produce lower return velocities for the sampled returning trajectories. This makes the effect of foot placement easy to compare and allows the walking controller to use a lookup table indexed by the current midstance velocity.



## Control as a Lookup Table

Now we want to ask the question: given how fast the walker is moving now, where should it place its next foot so that it can eventually stop?

The lookup-table controller answers this by first simulating individual foot-placement choices, then connecting their outcomes into routes to the balancing RoA.

To build the lookup-table, I first discretized the midstance velocities.
Since the Poincaré section I chose was the positive $\dot{\theta}$ axis of the phase portrait, $\theta=0$ and $\dot{\theta}>0$, so only angular velocity needed to be sampled. The initial grid contains 11 velocities from $0$ to $\sqrt{2g/\ell}\approx4.43$ rad/s, with spacing approximately $0.443$ rad/s.

I then discretized the foot-placement choices. I sampled 5 values of $\alpha$ between $\pi/8$ and $\pi/7$. Then, combining these with the 11 velocities gave a table with 11 rows and 5 columns. Each row represents a starting midstance velocity, and each column represents an action. A cell asks what happens when the walker starts at $(0,\dot{\theta}_i)$ and uses the selected $\alpha_j$.

After this setup, I then simulated each state-action pair.
 Each trial was classified by whether it reached the estimated RoA, returned to midstance after a footstrike, failed, or reached the time limit. The table stores the outcome, the number of footstrikes, and the next midstance velocity when there is a return. RoA entry is checked along the trajectory, so the walker can be captured before the next midstance crossing.



The state–action table below is a visualization of this initial $11\times5$ lookup table.

<img src="../output/assignment_2/return_table.png" alt="" width="700">


It shows which foot-placement choices reach the balancing RoA, permit continued walking, or fail at each sampled speed. Together with the stored return velocities and footstrike counts, it provides the transitions used to plan a sequence of footsteps toward the balancing RoA.

Once the table is built, it was possible to work backward from states that reach the RoA. A state already inside the RoA needs zero steps. If one footstrike brings it into the RoA, it needs one step. If one footstrike brings it to a state that needs one more step, it needs two.

This process is repeated to find the fewest predicted footsteps from each sampled velocity, and then the count is stored in minimum_steps and the first foot-placement angle in best_alpha. When a return velocity falls between grid points, the nearest sampled velocity is used to look up the remaining steps.

I then use this policy during a walking simulation. At each midstance crossing, the controller finds the sampled velocity closest to the actual velocity and chooses its corresponding best_alpha. For example, with the 11-point grid, an actual velocity of $1.2$ rad/s uses the action stored at approximately $1.329$ rad/s. The simulation still continues from the actual velocity of $1.2$ rad/s. Repeating this lookup at each crossing produces a sequence of foot placements, until the walker enters the RoA and the ankle controller takes over.


Plotting minimum_steps against the initial midstance velocity shows which sampled states have predicted one-step, two-step, and longer routes to the balancing RoA. The plot below uses the final $641\times5$ lookup table, selected using the grid-resolution check described towards the end of the report.

<img src="../output/assignment_2/final_grid/minimum_steps.png" alt="Predicted minimum footstrikes from the final 641-velocity lookup table" width="700">


## Walking trajectories

For both trajectories below, I used the selected $641\times5$ lookup table and the $33\times49$ RoA grid. Each simulation ran for about 10 s with a timestep of $0.001$ s, so the plots include both walking and the later balancing motion.

### Minimum-step trajectory

I first simulated the walker from the initial condition $(\theta_0,\dot{\theta}_0)=(0,3)$ using the minimum-step lookup-table policy. At each midstance crossing, the controller uses the nearest sampled velocity to choose the foot-placement angle with the fewest predicted footsteps remaining. The plots below show the angle and angular velocity recorded during this continuous simulation.

<img src="../output/assignment_2/final_grid/walking_trajectory.png" alt="" width="700">

In this simulation, the walker reached the estimated RoA after three footstrikes, matching the table's prediction. The gray dashed lines mark the simulated footstrikes, and the green line marks RoA entry. At that point, the ankle controller takes over and brings the angle and angular velocity toward zero. The trajectory satisfies the standing convergence criterion by the end of the simulation.

### Maximum-step trajectory

I also simulated the walker from the same initial condition, $(\theta_0,\dot{\theta}_0)=(0,3)$, using the maximum-step lookup-table policy. This policy chooses foot-placement actions that give the largest predicted number of footsteps before reaching the balancing RoA. It uses the same return table, but keeps the longest successful routes when choosing an action for each sampled velocity.

<img src="../output/assignment_2/final_grid/maximum_steps_trajectory.png" alt="" width="700">


In this simulation, the walker reached the estimated RoA after five footstrikes, compared with three under the minimum-step policy. This also matches the table's prediction. The gray dashed lines again mark footstrikes, and the green line marks RoA entry. The ankle controller then takes over, and the trajectory satisfies the same standing convergence criterion.

These results show how the two policies choose different routes from the same initial condition, using the same sampled foot-placement angles and estimated RoA.

## Grid Resolution

To see how grid resolution affects the lookup table, I compared tables with 11, 321, and 641 sampled velocities over the same range from $0$ to approximately $4.43$ rad/s. I kept the RoA grid at $33\times49$, used the same 5 foot-placement angles, and kept the timestep at $0.001$ s.

<img src="../output/assignment_2/grid_comparison/return_table_resolution.png" alt="Return tables with 11, 321, and 641 sampled velocities and the same five foot-placement angles" width="900">

The 11-point table misses the narrow green band near $1.2$ rad/s at the largest $\alpha$. The 321- and 641-point tables both show this band and look similar, but their footstep predictions can still differ.

When predicting footsteps, I replace each return velocity with the nearest grid value. In the walking simulation, I use that grid value to choose the next foot-placement angle, but the walker continues from its actual velocity. Even a small difference between the actual and sampled velocities can change the outcome: the table may predict that the next step reaches the RoA, while the simulated walker needs another step.

To check this, I used the same 13 starting velocities for each grid, including $3$ rad/s and values between grid points. I tested both policies, giving 26 trials per grid. Each simulation ran for 10 s. I compared the predicted footstrike count with the count before RoA entry, and checked that $|\theta|<10^{-3}$ rad and $|\dot{\theta}|<10^{-3}$ rad/s throughout the final 0.5 s. A grid had to match every count and balance in every trial, with the same results on finer tested grids.

| Sampled velocities | Predictions matching the simulation |
| --- | --- |
| 321 | 25/26 |
| 641 | 26/26 |
| 1281 | 26/26 |

All trials reached standing. At approximately $4.027$ rad/s, the 321-point grid predicted four footstrikes under the minimum-step policy, but the simulation took five. The 641- and 1281-point grids matched all 26 predictions and gave identical observed counts. I therefore selected the $641\times5$ table as the coarsest tested grid passing this check for these initial velocities, with the RoA and control angles held fixed.


## Conclusions

This assignment showed how discrete foot placement and continuous ankle control can work together to bring the walker to standing. The foot-placement controller guides the walker into the ankle controller’s RoA, where ankle torque then stabilizes the upright position. Choosing midstance as the Poincaré section reduced the walking state to a single angular velocity, making it possible to build a lookup table and work backward from successful outcomes to choose actions.

I also learned that grid resolution affects the accuracy of the predicted footstep counts. Two return tables can look similar while giving different predictions near the boundaries between outcomes. Comparing these predictions with full walking simulations helped justify selecting the $641\times5$ table. From the same initial condition, $(0,3)$, the minimum-step policy reached the RoA after three footstrikes, while the maximum-step policy took five. Both trajectories then settled to standing, showing how different foot-placement choices can produce different routes to the same equilibrium.