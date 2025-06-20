from typing import Tuple
from copy import deepcopy

import numpy as np
from enact import nmat_measure
from enlib import fft, enmap
from enlib import pmat
from pixell import enplot

from .tod import TOD

def create_map(tod: TOD, coord: Tuple[float, float], geometry: Tuple[float, Tuple[int, int]]) -> None:
    """
    Generate a polarization map (T, Q, U) from telescope time-ordered data.

    Parameters:
        tod (TOD): Telescope time-ordered data.
        coord (tuple): (ra, dec) coordinates for the map center.
        geometry (tuple): (res, pixels) where res is the map resolution and
                          pixels is a tuple (x_pixels, y_pixels).
    """

    # Initialize file database and load the relevant data.
    
    data = deepcopy(tod.data)
    scan = tod.scan

    # Estimate noise using a Fourier transform of the TOD.
    ft = fft.rfft(data) * tod.num_samples**(-1./2.)
    noise = nmat_measure.detvecs_jon(ft, scan.srate)

    # Unpack coordinate and geometry settings.
    ra, dec = coord  # enmap.geometry expects (dec, ra) order below.
    res, pixels = geometry
    shape, wcs = enmap.geometry((dec, ra), res, pixels)

    # Allocate a three-channel map (T, Q, U).
    omap = enmap.zeros((3,) + shape, wcs, np.float32)

    # Apply the noise characteristics to the TOD.
    noise.apply(data)

    # Project the TOD onto the spatial map using the pointing matrix.
    P = pmat.PmatMap(scan, omap)
    P.backward(data.astype(np.float32), omap)

    return omap

# TODO: Allow for other plotting options coming from the instrument API.
def plot_map(omap: np.ndarray, component: int = 0) -> None:
    """
    Display the temperature channel of a polarization map.

    Parameters:
        omap (np.ndarray): A three-channel (T, Q, U) map.
        component (int): The map channel to plot. Default is 0 (temperature).
    """

    enplot.pshow(omap[component], autocrop=False, mask=0, quantile=0)
