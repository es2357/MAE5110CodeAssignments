def integrate(t, state, timestep, dynamics, params):
    return state + timestep * dynamics(t, state, params)