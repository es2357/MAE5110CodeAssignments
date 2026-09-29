import numpy as np

def generate_params():
    params = {
        "gravity": 9.81,
        "mass": 1.0,
        "spoke_length": 1.0,
        "gamma": 0.2,
        "num_spokes": 6,
        "alpha": np.pi/6,
    }
    return params

def dynamics(t, state, params):
    gravity = params["gravity"]
    mass = params["mass"]
    spoke_length = params["spoke_length"]
    gamma = params["gamma"]
    alpha = params["alpha"]
    num_spokes = params["num_spokes"]
    
    theta = state[0]
    angular_velocity = state[1]
    
    angular_acceleration = (gravity/spoke_length) * np.sin(theta)
    
    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative
    
def detect_forward_event(t, state, params):
    gamma = params['gamma']
    alpha = params['alpha']
    return state[0] - (gamma + alpha)
detect_forward_event.terminal = True
detect_forward_event.direction = 1

def detect_backward_event(t, state, params):
    gamma = params['gamma']
    alpha = params['alpha']
    return state[0] - (gamma - alpha)
detect_backward_event.terminal = True
detect_backward_event.direction = -1

def reset_map(state, params):
    
    alpha = params["alpha"]
    
    theta =state[0]
    angular_velocity = state[1]
    
    theta_new = theta - 2 * alpha
    angular_velocity_new = angular_velocity * np.cos(2 * alpha)
    return np.array([theta_new, angular_velocity_new])

def energy_calc(state, params):
    
    gravity = params["gravity"]
    mass = params["mass"]
    spoke_length = params["spoke_length"]
    
    theta = state[0]
    angular_velocity = state[1]
    
    potential_energy = mass * gravity * spoke_length * np.cos(theta)
    kinetic_energy = 0.5 * mass * spoke_length**2 * angular_velocity**2
    return potential_energy, kinetic_energy
