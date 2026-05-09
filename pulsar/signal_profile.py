from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .target import Target
from .timing import TimingModel, PeriodTimingModel

class SignalProfile(ABC):
    """
    Abstract base class for pulsar signal profiles.
    """
    target: Target
    timing: Optional[TimingModel] = None

    def __init__(self, target: Target, timing: Optional[TimingModel] = None):
        self.target = target
        self.timing = timing or PeriodTimingModel(target)

    def __str__(self) -> str:
        return f'Signal Profile: {self.name}, {self.target}'

    @property
    @abstractmethod
    def name(self): # pragma: no cover
        """
        Abstract property to return the name of the profile.
        """
        pass

    @abstractmethod
    def profile(self, phases): # pragma: no cover
        """
        Abstract method to compute the pulsar profile for given phases.

        Parameters:
        - phases ([float]): An array of phase values.

        Returns:
        - [float]: The pulsar profile corresponding to the given phases.
        """
        pass

    @property
    def src(self):
        t = self.target
        if t is None:
            return None
        return np.array([t.ra, t.dec, t.amp], dtype = float)

    @property
    def phase_exclusive(self) -> bool:
        """
        Indicates whether the profile is phase-exclusive, meaning it cannot 
        be processed simultaneously with other profiles due to phase overlap.
        
        Profiles like von Mises extend beyond their central phase bin, requiring 
        individual flux estimation to prevent interference. In contrast, boxcar 
        profiles are evenly spaced and can be estimated simultaneously.
        
        Returns:
            bool: True if the profile requires individual estimation, False if it can 
                be processed with others.
        """
        return False 

    def obstime2phase(self, ctime: float) -> float:
        """
        Calculate the phase of the pulsar at a given observational time.

        Parameters:
        - ctime (float): Current observational time.

        Returns:
        - float: Phase of the pulsar.
        """
        return self.timing.obstime2phase(ctime)

    def obstime2profile(self, ctime: float) -> float:
        """
        Compute the pulsar's emission profile at a given observational time.

        Parameters:
        - ctime (float): Current observational time.

        Returns:
        - float: Value of the profile function at the calculated phase.
        """
        phase = self.obstime2phase(ctime)
        return self.profile(phase)
    
    def obstime2flux(self, ctime: float) -> float:
        """
        Calculate the flux (brightness) of the pulsar at a given observational time.

        Parameters:
        - ctime (float): Current observational time.

        Returns:
        - float: Computed flux based on the profile and source amplitude.
        """
        phase = self.obstime2phase(ctime)
        amp = self.target.amp if self.target is not None else 1
        return amp * self.profile(phase)

    def as_dict(self) -> dict:
        """
        Convert the profile to a dictionary.

        Returns:
        - dict: Dictionary representation of the profile.
        """
        result = {
            'name': self.name,
            'target': self.target.as_dict() if self.target else None,
            'timing': str(self.timing) if self.timing else None,
        }

        # Add any extra subclass-specific attributes
        extra_attrs = {
            k: v for k, v in self.__dict__.items()
            if k not in result and not k.startswith('_')
        }
        result.update(extra_attrs)

        return result


class RemovalProfile(SignalProfile):
    """
    Sentinel profile used to mark a source for in-place removal from the TOD
    (via `tod.remove_source` inside the search). The profile itself returns
    zeros so it never contributes injected signal — only its target position
    matters.
    """
    def __init__(self, target=None, timing: Optional[TimingModel] = None):
        super().__init__(target=target, timing=timing)

    @property
    def name(self):
        return 'Removal'

    def profile(self, phi):
        return np.zeros_like(phi)

class ConstantProfile(SignalProfile):
    """
    Implementation of SignalProfile that returns a constant profile.
    """

    def __init__(self, target=None, timing: Optional[TimingModel] = None):
        super().__init__(target=target, timing=timing)

    @property
    def name(self):
        return 'Constant'
    
    def profile(self, phi):
        return np.ones_like(phi)

@dataclass
class VonMisesProfile(SignalProfile):
    """
    Implementation of SignalProfile that models the profile using a von Mises distribution.
    """

    def __init__(self, target=None, timing: Optional[TimingModel] = None, mu: float = 1.0):
        super().__init__(target=target, timing=timing)
        self.mu = mu

    @property
    def name(self):
        return 'Von Mises'

    @property
    def phase_exclusive(self) -> bool:
        return True
    
    def profile(self, phi):
        kappa = np.log(2) / (2 * (np.sin(np.pi * self.target.D / 2) ** 2))
        return np.exp(kappa * (np.cos(phi * 2 * np.pi) - self.mu))

@dataclass
class BoxcarProfile(SignalProfile):
    """
    Implementation of a boxcar function, which models a profile as a discrete bin
    within a cycle divided into a specified number of bins.
    """

    def __init__(self, target=None, timing: Optional[TimingModel] = None,
                 num_bins: int =10, bin_index: int = 0):
        super().__init__(target=target, timing=timing)
        self.num_bins = num_bins
        self.bin_index = bin_index

    @property
    def name(self):
        return 'Boxcar'

    def profile(self, phase):
        """
        Compute the profile at a given phase.

        Parameters:
        - phase (float or np.ndarray): The phase value(s) ranging from 0 to 1.

        Returns:
        - np.ndarray: Array indicating where the phase hits the specified bin.
        """
        # Ensure phase is a numpy array for vectorized operations
        phase = np.asarray(phase)

        # Calculate indices from phase values
        idx = np.round(phase * self.num_bins).astype(int) % self.num_bins
        
        return (idx == self.bin_index).astype(int)

def create_boxcar_profiles(num_bins: int, target: Optional[Target] = None,
                           timing: Optional[TimingModel] = None) -> List[SignalProfile]:
    """
    Create a list of boxcar profiles with the specified number of bins.

    Parameters:
    - num_bins (int): Number of bins in each boxcar profile.
    - target (Target): Target object containing the boxcar parameters.
    - timing (TimingModel): Timing model object for the boxcar profiles.

    Returns:
    - list: List of boxcar profiles.
    """
    return [BoxcarProfile(num_bins=num_bins, bin_index=i, target=target, timing=timing) for i in range(num_bins)]

def create_von_mises_profiles(n: int, target: Target, timing: Optional[TimingModel] = None,
                              target_phase: float = 0) -> List[SignalProfile]:
    """
    Create a list of von Mises profiles with the specified number of profiles,
    changing the initial phase of a copied target.

    Parameters:
    - n (int): Number of von Mises profiles to create.
    - target (Target): Target object containing the von Mises parameters to copy.
    - timing (TimingModel): Timing model object for the von Mises profiles.
    - target_phase (float): Initial phase of the copied target. Default is 0.

    Returns:
    - list: List of von Mises profiles.
    """
    phases = np.linspace(0, 1, n, endpoint=False)
    shifted_phases = (phases + target_phase) % 1

    profiles = []
    for phase in shifted_phases:
        t = copy(target)
        t.phi0 = phase
        profiles.append(VonMisesProfile(target=t, timing=timing))
    return profiles