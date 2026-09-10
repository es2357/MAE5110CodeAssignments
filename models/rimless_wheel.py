import numpy as np

# RIMLESS WHEEL MODEL

def generate_params():
    return{
        "gravity": 9.81,  # gravity m/s^2)
        "length": 1,  # rod length (m)
        "mass": 0.2,  # point at center
        "num_spokes": 8,  # number of spokes
        "slope_angle": np.pi / 6,  # slope angle (rad)
    }
