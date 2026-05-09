from dataclasses import dataclass, asdict
from typing import Optional, List

import numpy as np
from scipy.signal import butter, filtfilt
from pixell import fft

from .target import Source
from .tod import TOD

@dataclass
class FilterOptions:
    """
    Bundles parameters for the supported high-pass filters so a single
    `FilterOptions` instance can travel through Scenario → Storage and be
    serialised alongside results for reproducibility.

    Only the fields relevant to the chosen `method` are read; the rest are ignored.

    Attributes:
    - method (str): One of 'butterworth', 'fft', or 'planet'.
        - 'butterworth': time-domain IIR high-pass; pick a sharp cutoff with
          predictable phase behaviour.
        - 'fft': frequency-domain attenuation with a tunable roll-off; cheaper
          and more flexible than Butterworth for steep slopes.
        - 'planet': removes a sky region (RA, Dec, radius) before applying an
          FFT-style filter to the localised model; used to subtract bright
          sources without filtering the whole TOD.
    - cutoff (float): Cutoff frequency for the Butterworth filter, in Hz.
    - order (int): Order of the Butterworth filter; higher = steeper roll-off.
    - fknee (float): Knee frequency for the FFT/planet filter, in Hz.
    - alpha (float): Slope of the FFT/planet filter; higher = steeper attenuation.
    """
    method: str = ''
    cutoff: float = 0.0
    order: int = 0
    fknee: float = 0.0
    alpha: float = 0.0

    def __str__(self):
        """
        Returns a string representation of the FilterOptions.
        """
        return (f"FilterOptions:\n"
                f"  Method: {self.method}\n"
                f"  Cutoff Frequency: {self.cutoff} Hz\n"
                f"  Order: {self.order}\n"
                f"  Knee Frequency: {self.fknee} Hz\n"
                f"  Alpha: {self.alpha}")
    
    def as_dict(self):
        """
        Serializes the FilterOptions object to a dictionary.
        """
        return asdict(self)

def filter(tod: TOD, opt: FilterOptions, sources: Optional[List[Source]] = None):
    """
    Dispatch to the high-pass filter selected by `opt.method`. Centralising the
    dispatch lets Scenario carry a single FilterOptions instance regardless of
    which method is chosen.

    Parameters:
    - tod (TOD): The time-ordered object containing the data and scan information.
    - opt (FilterOptions): The options for the filter (see FilterOptions).
    - sources (List[Source]): Required only for the 'planet' method — the
        sky regions whose contribution should be modelled and subtracted.
    """
    if opt.method == 'butterworth':
        highpass_filter_butterworth(tod, opt.cutoff, opt.order)
    elif opt.method == 'fft':
        highpass_filter_fft(tod, opt.fknee, opt.alpha)
    elif opt.method == 'planet':
        if sources is None:
            raise ValueError("At least one source object must be provided for planet filter.")
        for source in sources:
            planet_filter(tod, source.ra, source.dec, source.radius, opt.fknee, opt.alpha)
    else:
        raise ValueError(f"Invalid filter method: {opt.method}")

def highpass_filter_butterworth(tod: TOD, cutoff: float, order: float=1):
    """
    Apply a high-pass filter to the data using a Butterworth filter designed via scipy's butter function.

    Parameters:
    - tod (TOD): The time-ordered object containing the data and scan information.
    - cutoff (float): The cutoff frequency of the high-pass filter (in Hz).
    - order (int): The order of the filter (default is 1).
    """
    fs = tod.sampling_rate

    b, a = butter(order, cutoff, btype='highpass', analog=False, fs=fs, output='ba')
    tod.data = filtfilt(b, a, tod.data, method='gust')

def highpass_filter_fft(tod: TOD, fknee: float, alpha: float):
    """
    Apply a high-pass filter directly in the frequency domain using FFT.

    Parameters:
    - tod (TOD): The time-ordered object containing the data and scan information.
    - fknee (float): The knee frequency of the filter (in Hz) where the filter starts to attenuate.
    - alpha (float): The slope of the filter, controlling how quickly the filter attenuates the signal.
                     A higher alpha results in a steeper attenuation.
    """
    
    fs = tod.sampling_rate
    samples = tod.num_samples

    freq = fft.rfftfreq(samples, 1/fs)
    ftod = fft.rfft(tod.data)
    filter_curve = 1 + (np.maximum(freq, freq[1]/2) / fknee) ** -alpha
    ftod /= filter_curve
    tod.data = fft.irfft(ftod, tod.data, n=samples, normalize=True)

def planet_filter(tod: TOD, ra: float, dec: float, R: float, fknee: float,
                  alpha: float):
    """
    Filter out celestial body interference from time-ordered data using a model
    and optional high-pass filter.

    Parameters:
    - tod (TOD): The time-ordered object from which the source's influence is to be removed.
    - ra (float): The right ascension (RA) (in rad) of the celestial source to be filtered out.
    - dec (float): The declination (Dec) (in rad) of the celestial source to be filtered out.
    - R (float): Radius (in rad) around the celestial source within which data is considered influenced by the source.
    - fknee (float): The knee frequency for the optional high-pass filter.
    - alpha (float): The slope of the optional high-pass filter.
    """

    fs = tod.sampling_rate
    samples = tod.num_samples
    model, _ = tod.locate_source(ra, dec, R)

    freq = fft.rfftfreq(samples, 1/fs)  # Compute frequency bins for the FFT
    ftod = fft.rfft(model)  # Compute the real FFT of the TOD
    filter_curve = 1 / (1 + (freq / fknee) ** alpha)  # Create the filter curve
    ftod *= filter_curve  # Apply the filter by multiplying in the frequency domain
    model = fft.irfft(ftod, model, n=samples, normalize=True)  # Transform back to time domain
    
    tod.data -= model