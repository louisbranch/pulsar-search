from copy import copy
import numpy as np
import pytest
from scipy import fft
from .test_mocks import create_mock_tod
from unittest.mock import patch

from pulsar import filter, FilterOptions, highpass_filter_butterworth, \
    highpass_filter_fft, planet_filter
from .fixtures import *

def generate_sine_wave(freq, sampling_rate=400, duration=1):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

class TestFilterOptions:

    def test_str(self):
        options = FilterOptions(method='butterworth', cutoff=5.0, order=3, fknee=0.0, alpha=0.0)
        expected_str = ("FilterOptions:\n  Method: butterworth\n  Cutoff Frequency: 5.0 Hz\n"
                        "  Order: 3\n  Knee Frequency: 0.0 Hz\n  Alpha: 0.0")
        assert str(options) == expected_str

    def test_as_dict(self):
        options = FilterOptions(method='butterworth', cutoff=5.0, order=3, fknee=0.0, alpha=0.0)
        expected_dict = {
            'method': 'butterworth',
            'cutoff': 5.0,
            'order': 3,
            'fknee': 0.0,
            'alpha': 0.0,
        }
        assert options.as_dict() == expected_dict

def test_filter(target):
    signal = generate_sine_wave(30, 400)
    tod = create_mock_tod()
    tod.data = signal
    tod.num_samples = len(signal)

    options = FilterOptions(method='butterworth', cutoff=2, order=1)
    filter(tod, options)

    options = FilterOptions(method='fft', fknee=5, alpha=10)
    filter(tod, options)

    options = FilterOptions(method='planet', fknee=5, alpha=10)
    target.radius = 0.2
    tod.locate_source.side_effect = lambda ra, dec, R: (copy(signal), np.ones(len(signal)))
    filter(tod, options, sources=[target])

    with pytest.raises(ValueError):
        filter(tod, options)

    with pytest.raises(ValueError):
        options = FilterOptions(method='invalid', fknee=5, alpha=10)
        filter(tod, options, sources=[target])

def test_filter_planet_with_multiple_sources(target):
    # The planet branch loops over every source. Use a counting side_effect to
    # confirm the loop runs once per source.
    from pulsar import Source
    signal = generate_sine_wave(30, 400)
    tod = create_mock_tod()
    tod.data = copy(signal)
    tod.num_samples = len(signal)
    tod.locate_source.side_effect = lambda ra, dec, R: (copy(signal), np.ones(len(signal)))

    s1 = Source(name='s1', ra=0.1, dec=0.2, radius=0.05)
    s2 = Source(name='s2', ra=0.3, dec=0.4, radius=0.05)

    options = FilterOptions(method='planet', fknee=5, alpha=10)
    filter(tod, options, sources=[s1, s2])

    assert tod.locate_source.call_count == 2

def test_highpass_filter_butterworth():

    fs = 400  # Sampling frequency (Hz)
    low_freq = 2  # Hz
    high_freq = 30  # Hz
    cutoff = 5  # Cutoff frequency (Hz)
    order = 5  # Filter order

    low_freq_signal = generate_sine_wave(low_freq, fs)
    high_freq_signal = generate_sine_wave(high_freq, fs)

    low_freq_tod = create_mock_tod()
    low_freq_tod.data = low_freq_signal
    low_freq_tod.num_samples = len(low_freq_signal)

    high_freq_tod = create_mock_tod()
    high_freq_tod.data = high_freq_signal
    high_freq_tod.num_samples = len(high_freq_signal)

    highpass_filter_butterworth(low_freq_tod, cutoff, order)
    highpass_filter_butterworth(high_freq_tod, cutoff, order)

    # FFT of original and filtered signals
    freqs = fft.rfftfreq(len(low_freq_signal), 1/fs)
    orig_low_fft = np.abs(fft.rfft(low_freq_signal))
    filt_low_butter_fft = np.abs(fft.rfft(low_freq_tod.data))

    orig_high_fft = np.abs(fft.rfft(high_freq_signal))
    filt_high_butter_fft = np.abs(fft.rfft(high_freq_tod.data))

    # Assertions for low frequencies (0 to cutoff)
    low_freq_range = freqs <= cutoff
    assert np.sum(filt_low_butter_fft[low_freq_range]) < 0.5 * np.sum(orig_low_fft[low_freq_range])

    # Assertions for high frequencies (above cutoff)
    high_freq_range = freqs > cutoff
    assert pytest.approx(np.sum(filt_high_butter_fft[high_freq_range]), 4) == np.sum(orig_high_fft[high_freq_range])

def test_highpass_filter_fft():

    fs = 400  # Sampling frequency (Hz)
    low_freq = 2  # Hz
    high_freq = 30  # Hz
    fknee = 5 # Knee frequency (Hz)
    alpha = 10

    low_freq_signal = generate_sine_wave(low_freq, fs)
    high_freq_signal = generate_sine_wave(high_freq, fs)

    low_freq_tod = create_mock_tod()
    low_freq_tod.data = copy(low_freq_signal)
    low_freq_tod.num_samples = len(low_freq_signal)

    high_freq_tod = create_mock_tod()
    high_freq_tod.data = copy(high_freq_signal)
    high_freq_tod.num_samples = len(high_freq_signal)

    highpass_filter_fft(low_freq_tod, fknee, alpha)
    highpass_filter_fft(high_freq_tod, fknee, alpha)

    # FFT of original and filtered signals
    freqs = fft.rfftfreq(len(low_freq_signal), 1/fs)
    orig_low_fft = np.abs(fft.rfft(low_freq_signal))
    filt_low_fft_fft = np.abs(fft.rfft(low_freq_tod.data))

    orig_high_fft = np.abs(fft.rfft(high_freq_signal))
    filt_high_fft_fft = np.abs(fft.rfft(high_freq_tod.data))

    # Assertions for low frequencies (0 to cutoff)
    low_freq_range = freqs <= fknee
    assert np.sum(filt_low_fft_fft[low_freq_range]) < 0.5 * np.sum(orig_low_fft[low_freq_range])

    # Assertions for high frequencies (above cutoff)
    high_freq_range = freqs > fknee
    assert pytest.approx(np.sum(filt_high_fft_fft[high_freq_range]), 0.01) == np.sum(orig_high_fft[high_freq_range])

def test_planet_filter():

    fs = 400  # Sampling frequency (Hz)
    low_freq = 2  # Hz
    high_freq = 30  # Hz

    low_freq_signal = generate_sine_wave(low_freq, fs)
    high_freq_signal = generate_sine_wave(high_freq, fs)

    low_freq_tod = create_mock_tod()
    low_freq_tod.data = copy(low_freq_signal)
    low_freq_tod.num_samples = len(low_freq_signal)
    low_freq_tod.locate_source.side_effect = lambda ra, dec, R: (copy(low_freq_signal), np.ones(len(low_freq_signal)))

    high_freq_tod = create_mock_tod()
    high_freq_tod.data = copy(high_freq_signal)
    high_freq_tod.num_samples = len(high_freq_signal)
    high_freq_tod.locate_source.side_effect = lambda ra, dec, R: (copy(high_freq_signal), np.ones(len(high_freq_signal)))

    planet_filter(low_freq_tod, ra=0, dec=0, R=0, fknee=5, alpha=10)
    planet_filter(high_freq_tod, ra=0, dec=0, R=0, fknee=5, alpha=10)

    # FFT of original and filtered signals
    freqs = fft.rfftfreq(len(low_freq_signal), 1/fs)
    orig_low_fft = np.abs(fft.rfft(low_freq_signal))
    filt_low_planet_fft = np.abs(fft.rfft(low_freq_tod.data))

    orig_high_fft = np.abs(fft.rfft(high_freq_signal))
    filt_high_planet_fft = np.abs(fft.rfft(high_freq_tod.data))

    # Assertions for low frequencies (0 to cutoff)
    low_freq_range = freqs <= 5
    assert np.sum(filt_low_planet_fft[low_freq_range]) < 0.5 * np.sum(orig_low_fft[low_freq_range])

    # Assertions for high frequencies (above cutoff)
    high_freq_range = freqs > 5
    assert pytest.approx(np.sum(filt_high_planet_fft[high_freq_range]), 0.01) == np.sum(orig_high_fft[high_freq_range])