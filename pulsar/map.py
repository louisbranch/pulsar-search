from typing import Tuple

import numpy as np

from .tod import TOD
from .config import config

def create_map(tod: TOD, coord: Tuple[float, float], 
               geometry: Tuple[float, Tuple[int, int]]) -> None: # pragma: no cover
    """
    Generate a polarization map (T, Q, U) from telescope time-ordered data.

    Parameters:
        tod (TOD): Telescope time-ordered data.
        coord (tuple): (ra, dec) coordinates for the map center.
        geometry (tuple): (res, pixels) where res is the map resolution and
                          pixels is a tuple (x_pixels, y_pixels).
    """

    return config.instrument.create_map(tod, coord, geometry)

def plot_map(omap: np.ndarray, component: int = 0) -> None:
    """
    Display the temperature channel of a polarization map.

    Parameters:
        omap (np.ndarray): A three-channel (T, Q, U) map.
        component (int): The map channel to plot. Default is 0 (temperature).
    """

    config.instrument.plot_map(omap, component)
