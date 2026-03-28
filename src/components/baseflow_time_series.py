import numpy as np
from components.baseflow import evaluate_baseflow

def construct_baseflow_time_series(profile_name, z, t_grid, U_time_modulation=None):
    """
    Construct U(z, t) for the base flow. If U_time_modulation is None, U(z, t) = U(z) for all t.
    Optionally, U_time_modulation(t) can be a function to modulate U(z) in time.
    Returns Uzt of shape (len(z), len(t_grid)).
    """
    U = evaluate_baseflow(profile_name, z)["U"].real  # shape (len(z),)
    if U_time_modulation is None:
        Uzt = np.tile(U[:, None], (1, len(t_grid)))
    else:
        Uzt = np.zeros((len(z), len(t_grid)))
        for i, t in enumerate(t_grid):
            Uzt[:, i] = U * U_time_modulation(t)
    return Uzt
