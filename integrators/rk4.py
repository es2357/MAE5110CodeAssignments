

def step(dynamics, t, state, timestep, params):
    """Advance a dynamical system by one fourth-order Runge–Kutta step."""
    h = timestep

    k1 = dynamics(t, state, params)

    k2 = dynamics(
        t + h / 2,
        state + h * k1 / 2,
        params,
    )

    k3 = dynamics(
        t + h / 2,
        state + h * k2 / 2,
        params,
    )

    k4 = dynamics(
        t + h,
        state + h * k3,
        params,
    )

    next_state = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return next_state