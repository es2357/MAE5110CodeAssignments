

def step(dynamics, t, state, timestep, params):
    """Advance a dynamical system by one Explicit Euler step"""
    state_deriv = dynamics(t, state, params)
    next_state = state + timestep * state_deriv
    return next_state