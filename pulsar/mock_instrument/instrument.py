from dataclasses import dataclass, field
from typing import List, Optional, Generator, Tuple

import numpy as np
from numpy.random import default_rng

from pulsar import TOD
from pulsar.pointing_models import PointingModel

@dataclass
class Config:
    freq_range: Tuple[float, float] = (1.0, 100.0)  # Frequency range in Hz
    one_over_f_noise: float = 3000.0  # Amplitude of 1/f noise
    gaussian_noise_std: float = 50.0  # Standard deviation of Gaussian noise
    amplitude_range: Tuple[float, float] = (0.5, 1.5)  # Min, max amplitude for scaling
    trend_amplitude: float = 40000.0  # Amplitude of the atmospheric trend
    trend_frequency: float = 1e-4  # Frequency of the atmospheric trend (in Hz)
    trend_phase: float = 0.5  # Phase of the trend wave

@dataclass
class MockInstrument:
    """
    A mock instrument that implements the pulsar.Instrument protocol.
    Generates synthetic TODs with configurable noise and an atmospheric trend
    so the search pipeline can run end-to-end without telescope data.
    """
    max_tods: int = 10
    dets_per_tod: int = 10
    samples_per_tod: int = 360_000
    sampling_rate: float = 400.0 # Hz
    tod_config: Config = field(default_factory=Config)
    rng: np.random.Generator = field(default_factory=default_rng)

    def tods(self, path: str, limit: int = 0, dets: Optional[List[int]] = None,
             ids: Optional[List[str]] = None) -> Generator[TOD, None, None]:
        count = 0
        for i in range(self.max_tods):
            id = str(i)
            if ids and id not in ids:
                continue

            num_detectors = self.dets_per_tod
            if dets:
                num_detectors = len(dets)

            yield TOD(
                id=id,
                num_detectors=num_detectors,
                num_samples=self.samples_per_tod,
                sampling_rate=self.sampling_rate,
                config=self.tod_config,
                rng=self.rng
            )

            count += 1
            if limit != 0 and count >= limit:
                break

        return iter([])

    def tod_ids(self, path: str) -> List[str]:
        return [str(i) for i in range(self.max_tods)]

    def pointing_model(self, *args, **kwargs) -> PointingModel:
        return None

@dataclass
class TOD:
    """
    A mock Time-Ordered Data (TOD) object that implements the pulsar.TOD protocol.
    """

    id: str
    num_detectors: int
    num_samples: int
    sampling_rate: float
    rng: np.random.Generator = field(default_factory=default_rng)
    calibrated: bool = True
    config: Config = field(default_factory=Config)
    _data: Optional[np.ndarray] = None

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            self._data = self._generate_signal()

        return self._data
    
    @data.setter
    def data(self, data: np.ndarray) -> None:
        self._data = data

    def _generate_signal(self) -> np.ndarray:
        config = self.config
        time = np.arange(self.num_samples) / self.sampling_rate
        
        # Create a base signal
        signal = np.zeros((self.num_detectors, self.num_samples))
        
        # Apply amplitude range
        amplitude = self.rng.uniform(*config.amplitude_range, size=(self.num_detectors, 1))
        signal += amplitude
        
        # Apply 1/f noise
        one_over_f = config.one_over_f_noise / np.maximum(1.0, np.abs(np.fft.fftfreq(self.num_samples)))
        one_over_f_noise = self.rng.normal(0, 1, size=(self.num_detectors, self.num_samples))
        one_over_f_noise = np.fft.ifft(np.fft.fft(one_over_f_noise) * one_over_f).real
        signal += one_over_f_noise
        
        # Apply Gaussian noise
        gaussian_noise = self.rng.normal(0, config.gaussian_noise_std, size=(self.num_detectors, self.num_samples))
        signal += gaussian_noise
        
        # Apply atmospheric trend
        trend = config.trend_amplitude * np.sin(2 * np.pi * config.trend_frequency * time + config.trend_phase)
        signal += trend
        
        return signal

    def calibrate(self) -> None:
        return

    def remove_source(self, ra: float, dec: float, R: float) -> None:
        return

    def locate_source(self, ra: float, dec: float, R: float) -> Tuple[np.ndarray, np.ndarray]:
        return

    def fill_gaps(self, data: np.ndarray = None):
        return