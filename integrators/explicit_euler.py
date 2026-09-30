# def integrate(t, state, timestep, dynamics, params):
def integrate(dynamics, t, state, timestep, params):
    return state + timestep * dynamics(t, state, params)