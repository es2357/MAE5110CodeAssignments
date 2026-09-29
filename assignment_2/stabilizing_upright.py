import matplotlib.pyplot as plt
import numpy as np
from integrators import rk4 as integrator
from models import inverted_pendulum_walker as model

def feedback_linearization(state, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]
    theta = state[0]
    angular_vel = state[1]
    Kd = 2.0
    Kp = 1.0
    
    torque_min = -0.1 * mass * gravity * length
    torque_max = 0.05 * mass * gravity * length
    
    gravity_term = -(gravity/length) * np.sin(theta) * mass * length ** 2
    stabilizing_term = mass * length ** 2 * (- Kd * angular_vel - Kp * theta)
    control_input = gravity_term + stabilizing_term
    
    return np.clip(control_input, torque_min, torque_max)

def is_upright(state, tolerance):
    theta = state[0]
    angular_vel = state[1]
    
    return ((np.abs(theta) < tolerance) & (np.abs(angular_vel) < tolerance))

def in_roa(state, theta_grid, angular_vel_grid, roa):
    theta = state[0]
    angular_vel = state[1]
    
    # State must be within strictly the sampled domain
    if not (
        theta_grid[0] < theta < theta_grid[-1] 
        and
        angular_vel_grid[0] < angular_vel < angular_vel_grid[-1]
    ):
        return False
    
    # Find the grid cell containing the state
    j_high_cell = np.searchsorted(theta_grid, theta)
    j_low_cell = j_high_cell - 1
    i_high_cell = np.searchsorted(angular_vel_grid, angular_vel)
    i_low_cell = i_high_cell - 1
    
    # Find adjacent grud values containing the state
    cell_values = [
        roa[i_low_cell, j_low_cell],
        roa[i_low_cell, j_high_cell],
        roa[i_high_cell, j_low_cell],
        roa[i_high_cell, j_high_cell]
    ]
    
    # Conservative RoA, all 4 adjacent points must be in the RoA as well.
    return bool(np.all(cell_values))

def roa_event_guard(previous_state, next_state, theta_grid, angular_vel_grid, roa):
    previous_inside = in_roa(previous_state, theta_grid, angular_vel_grid, roa)
    next_inside = in_roa(next_state, theta_grid,angular_vel_grid,roa)
    
    return (not previous_inside) and next_inside

def poincare_section(previous_state, next_state):
    previous_theta  = previous_state[0]
    next_theta = next_state[0]
    
    return(
        previous_theta < 0 and
        next_theta >= 0 and
        next_state[1] > 0
    )
    

# Verify ankle controller stabilizes
def ankle_controller_converges(initial_state, params):
    state = np.asarray(initial_state, dtype=float).copy()
    
    local_params = params.copy()
    alpha = local_params["angle_of_attack"]
    gamma = local_params["incline"]
    
    backward_guard = gamma - alpha
    forward_guard = gamma + alpha
    
    timestep = 1e-3
    max_time = 20.0
    tolerance = 1e-3
    
    t = 0.0
    settling_time = 0.5
    settling_steps = round(settling_time/timestep)
    settle_count = 0
    
    while t < max_time:
        local_params["ankle_torque"] = feedback_linearization(state, local_params)
        next_state = integrator.integrate(t, state, timestep, model.dynamics, local_params)
        
        if(
            next_state[0] <= backward_guard 
            or
            next_state[0] >= forward_guard
        ):
            return False
        
        if(is_upright(next_state, tolerance)):
            settle_count += 1
        else:
            settle_count = 0
        
        if settle_count >= settling_steps:
            return True
        
        state = next_state
        t += timestep
    return False

def is_valid_capture(state, theta_grid, angular_vel_grid, roa, params):
    if not in_roa(state, theta_grid, angular_vel_grid, roa):
        return False
    
    return ankle_controller_converges(state, params)