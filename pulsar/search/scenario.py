from dataclasses import asdict, dataclass
from enum import IntEnum
import hashlib
from typing import List, Optional, Tuple, Union

import numpy as np

from ..filters import FilterOptions
from ..noise import Noise
from ..target import Target, Source
from ..signal_profile import SignalProfile, RemovalProfile

Amplitude = Union[float, Tuple[float, float, float]]

@dataclass
class SignalOperation():
    """
    Class representing a signal operation to be applied to the TOD.

    Attributes:
    profile (SignalProfile): The signal profile of the operation.
    amp (Optional[Amplitude]): The amplitude of the operation. If None, the profile's target amplitude is used.
    """
    profile: SignalProfile = None
    amp: Optional[Amplitude] = None

    @property
    def position(self) -> Tuple[float, float, float]:
        return self.profile.target.ra, self.profile.target.dec, self.profile.target.radius

    def amplitude(self, ncomp: int) -> Amplitude:
        """
        Return the amplitudes as a 2D array with the correct number of components depending on the polarization.
        Exceeding amplitudes are ignored. Missing amplitudes are set to zero.
        """
        amp = self.amp
        if amp is None:
            amp = self.profile.target.amp

        amps = np.zeros((1, ncomp))

        if isinstance(amp, tuple):
            amps[0, :min(ncomp, len(amp))] = amp[:min(ncomp, 3)]
        else:
            amps[0, 0] = amp
        
        return amps

    @property
    def is_removal(self) -> bool:
        """
        Return True if the operation is a signal removal.
        """
        return isinstance(self.profile, RemovalProfile)

    def as_dict(self) -> dict:
        result = asdict(self)
        result['profile'] = self.profile.as_dict()
        return result

class PolarizationComponents(IntEnum):
    ONE = 1
    THREE = 3

@dataclass
class Scenario:
    """
    Scenario describing a set of target profiles and operations to be applied to list of TOD.
    Each scenario is uniquely identified by a seed, which is generated based on the scenario parameters.

    Attributes:
    title (str): Title of the scenario.
    target (Target): The target of the scenario. This is the default target for all profiles and operations.
    target_profiles (List[SignalProfile]): List of signal profiles to search for.
    operations (Optional[List[SignalOperation]]): List of signal operations to apply to the TOD, applied in order.
    calibration_options (Optional[dict]): Calibration options to apply to the TOD.
    filter (Optional[FilterOptions]): Filter to apply to the TOD.
    filter_sources (Optional[List[Source]]): List of sources to filter out.
    noise (Optional[Noise]): Noise to apply to the TOD.
    polarization_components (PolarizationComponents): Number of polarization components to use. Default is THREE.
    config (Optional[dict]): Additional configuration for the scenario.
    metadata (Optional[dict]): Additional metadata to store with the scenario.
    """

    title: str
    target: Target
    search_profiles: List[SignalProfile]
    operations: Optional[List[SignalOperation]] = None
    calibration_options: Optional[dict] = None
    filter: Optional[FilterOptions] = None
    filter_sources: Optional[List[Source]] = None
    noise: Optional[Noise] = None
    polarization_components: PolarizationComponents = PolarizationComponents.THREE
    config: Optional[dict] = None
    metadata: Optional[dict] = None
    seed: Optional[int] = None

    def __post_init__(self):
        if len(self.search_profiles) == 0:
            raise ValueError(f'At least one search profile must be provided for {self.title}.')

        if not isinstance(self.polarization_components, PolarizationComponents):
            raise ValueError(f'Invalid polarization components: {self.polarization_components}')
        
        if self.seed is None:
            self.seed = self._generate_seed()

    def _generate_seed(self) -> int:
        """
        Generate a pseudo-random seed based on the scenario parameters. This is used to ensure reproducibility.

        Parameters:
        fields (Any): The scenario fields to generate the seed from.

        Returns:
        int: The pseudo-random seed.
        """

        fields = self.as_dict()
        del fields['title']
        del fields['metadata']
        if self.config is None:
            del fields['config']

        fields['search_profiles'] = [p.as_dict() for p in self.search_profiles]

        attrs_str = repr(fields)
        hash_object = hashlib.sha256(attrs_str.encode())
        hash_int = int(hash_object.hexdigest(), 16)

        # Scale the hash to fit within the range of 0 to 2**32 - 1
        seed = hash_int % (2**32)
        self._seed = seed
        return seed

    def __str__(self) -> str:
        targets = len(self.search_profiles)
        ops = len(self.operations) if self.operations is not None else 0

        return f'{self.title} with {targets} target profiles with {ops} operations'

    def as_dict(self) -> dict:
        return asdict(self)
