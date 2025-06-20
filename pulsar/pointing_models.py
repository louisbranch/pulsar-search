from typing import Protocol, List

import numpy as np

from .tod import TOD
from .signal_profile import SignalProfile

class PointingModel(Protocol): # pragma: no cover

    def __init__(self, tod: TOD, sources: List[SignalProfile], ncomp: int=3): # pragma: no cover
        """
        Initializes the pointing model for a specific TOD and list of sources.

        Parameters:
            tod (TOD): The TOD object to which the pointing model is applied.
            sources (List[Profile]): The list of sources to be used for the pointing model.
            ncomp (int): The number of polarization components to use for the pointing.
                Default is 3: [T, Q, U].
        """
        pass

    def forward(self, data: np.ndarray, amps: np.ndarray, pmul: float=1): # pragma: no cover
        """
        Maps amplitudes to TOD.

        Parameters:
            data (np.ndarray): The TOD to be modified in-place.
            amps (np.ndarray): The amplitude parameters for each source and direction.
            pmul (float): A multiplicative factor for the amplitude parameters.
        """
        pass

    def backward(self, data: np.ndarray, pmul: float=1): # pragma: no cover
        """
        Extracts amplitudes from TOD.

        Parameters:
            data (np.ndarray): The TOD from which to extract amplitudes.
            pmul (float): A multiplicative factor for the amplitude parameters.
        """
        pass