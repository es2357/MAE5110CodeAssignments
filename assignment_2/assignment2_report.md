# Assignment 2
Thomas Istvan (ti237)
\
09/16/2026

### Initial Sketching

### Stabilizing with Feedback Linearization
Using the inverted pendulum dynamics with actuation we get the equation:
$$
\ddot{\theta} = \frac{g}{l}sin(\theta) + \frac{\tau}{ml^2}
$$
Applying feedback linearization to cancel out the non-linear term and introducing a desired control input u:
$$
\ddot{\theta} = u
$$
$$
u = \frac{g}{l}sin(\theta) + \frac{\tau}{ml^2}
$$
Isolating for the ankle torque:
$$
\tau = -ml^2\frac{g}{l}sin(\theta) - ml^2u
$$
$$
\tau = -mglsin(\theta) - ml^2u
$$
To stabilize the inverted pendulum in the upright equilibrium we can configure a control input u to add artifical damping and friction:
$$
u = -ml^2(K_d\dot{\theta} + K_p\theta)
$$
After feedback linearization, the dynamics are:
$$
\ddot{\theta} + K_d\dot{\theta} + K_p\theta = 0
$$
$$
\ddot{\theta} + 2\zeta{\omega_n}\dot{\theta} + {\omega_n}^2\theta = 0
$$
Relating the coefficents we can see that:
$$
K_d = 2\zeta{\omega_n}
$$
$$
K_p = {\omega_n}^2
$$

### Region of Attraction
The region of attraction (RoA) represents the set of initial states $(\theta, \dot{\theta})$ from which the ankle controller can stabilize the walker to the upright equilibrium without taking aother step. This was determined by numerically simulating the feedback linearization controllver over a grid of initial angles and velocities. An inital condition would be considered inisde the RoA if the controlled trajectroy converged to the upright equilibrium within some tolerance and remained there for 0.5 seconds. For the walking portion, the ankle torque is set to zero, but once the walking trajectory enters the RoA, the system can be stabilized at the upright equilibrium because of the controller.

### Poincaré  Section
For the Poincaré section, I selected the section to be at
$$
\theta = 0, \dot{\theta} > 0
$$
This section corresponds to when the walker is passing through the upright configuration while rotating in the forward walking direction (clockwise). Using this section reduces the state required for the walking lookup table from a full 2 dimensional state to only $\dot{\theta}$. Since $\theta$ is fixed at zero for every Poincaré crossing, the Poincaré map looks like:
$$
P(\dot{\theta_k}, \alpha_k) = \dot{\theta}_{k+1}
$$
For each Poincaré angular velocity and allowable angle of attack, the dynamics were simulated until the walker entered the RoA, failed, or went returned to the next Poincaré section. In order to generate the discretized dynamics while the trajectory is generated via the continuous dynamics.

### Control as a lookup table
The lookup-table resolution was verified by comparing lookup-table predictions with drect simulations at fixed set of  state-action pairs. Validation points were generated independently of the grids so that locations in between the lookup-grids could be tested. For each pair $(\dot{\theta}, \alpha)$ a direct simulation was used as the reference result. Where the nearest-neighbor lookup prediction was then compared with this reference result. 

To numerically classify my grid size as appropriate, I considered 2 types of error. First, I compared the transition classification, which is whether the trajectory entered the RoA, returned to the Poincaré section, or failed. Meaning, when a state-action pair is simulated there is a possible of 3 states. When those differ between the lookup table and the direct simulation, that is conisdered a transition classification error. Which is primarily caused from the look-up table method where error comes from the spacing between adjacent grid points and the use of the nearst-neighbor approximation. Secondly, I considered when both the direction simulation and lookup table returned to the Poincaré section, what is that error in the predicted angular velocity:
$$
error = |\dot{\theta^{lookup}}_{k+1} - \dot{\theta^{simulation}}_{k+1}|
$$
I evaluated the lookup tables with resolutions from 10 x 10 through 80 x 80. A resolution of 50 x 50 was selected because the transition behavior was shown to be substantially less sensitive to further grid refinment while the Poincaré error was also small. In this analysis, refinement from 40 x 40 to 50 x 50 changed the transition classification at 18 points, where refinement from 50 x 50 to 60 x 60 only changed by 3 points. The mean Poincaré error also decreased from 0.0220 rad/s to 0.0191 rad/s, the RMS error decreased from 0.0255 rad/s to 0.0217 rad/s, and the max error decreased from 0.0593 rad/s to 0.0456 rad/s. Any further grid refinement would cause longer computational time for a diminishing return in error.

Thus, showing that the 40 x 40 grid was still sensitive to refinement, especially in terms of transition classifications, while the changes became much smaller at a resolution of 50 x 50. Therefore, I utilized a 50 x 50 grid as the final lookup-table resolution.