from dataclasses import dataclass, field
import time
import functools
import hashlib
import os
from typing import Callable, Dict, List, Protocol, Optional
try:
    import psutil
except ImportError:
    psutil = None

import numpy as np
import matplotlib.pyplot as plt

from .scenario import Scenario
from ..tod import TOD
from .. import log

class Extension(Protocol):
    """
    Search extension interface for adding custom in-between steps to the search process.
    The extension should be registered with the search object to be used during the search.
    It should not modify the search state directly, but can be used to log additional information.
    """

    def before_step(self, step: str, scenario: Scenario, tod: TOD) -> None: # pragma: no cover
        """
        Triggered before each step in the search process.
        """
        pass

    def after_step(self, step: str, scenario: Scenario, tod: TOD) -> None: # pragma: no cover
        """
        Triggered after each step in the search process.
        """
        pass

class ExtensionManager:
    """
    Manager for search extensions. It allows registering extensions and notifying them before and after each step.
    Provides a decorator to wrap a step function with the before and after notifications.
    """

    def __init__(self):
        self.extensions: List[Extension] = []

    def register_extension(self, extension: Extension) -> None:
        self.extensions.append(extension)

    def notify_before(self, step: str, scenario: Scenario, tod: TOD) -> None:
        for ext in self.extensions:
            ext.before_step(step, scenario, tod)

    def notify_after(self, step: str, scenario: Scenario, tod: TOD) -> None:
        for ext in self.extensions:
            ext.after_step(step, scenario, tod)

    def wrap_step(self, step_name: str, scenario: Scenario, tod: TOD) -> Callable:
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.notify_before(step_name, scenario, tod)
                result = func(*args, **kwargs)
                self.notify_after(step_name, scenario, tod)
                return result
            return wrapper
        return decorator

@dataclass
class SamplingExtension(Extension):
    """
    Base extension that provides sampling behavior for derived extensions.
    It allows for a sampling rate to be set and uses a hash of the TOD id to determine if the TOD should be sampled.
    """
    sampling_rate: float = 0.1
    should_sample: bool = False
    steps: Optional[List[str]] = None
    detectors: Dict[str, int] = field(default_factory=dict)

    def should_sample_call(self, tod: TOD) -> None:
        # Use hashlib to generate a consistent hash value
        hash_object = hashlib.md5(tod.id.encode())
        hash_digest = hash_object.hexdigest()
        hash_int = int(hash_digest, 16)
        
        # Determine sampling based on hash value
        self.should_sample = (hash_int % 100) < (self.sampling_rate * 100)

        if tod.id in self.detectors:
            return self.detectors[tod.id]
        else:
            # Use the hash value to determine the detector index
            self.detectors[tod.id] = hash_int % tod.num_detectors

    def detector_index(self, tod: TOD) -> int:
        return self.detectors[tod.id]

    def before_step(self, step: str, scenario: Scenario, tod: TOD) -> None:
        if self.steps is None or step in self.steps:
            self.should_sample_call(tod)

    def after_step(self, step: str, scenario: Scenario, tod: TOD) -> None:
        if self.steps is None or step in self.steps:
            self.should_sample = False

@dataclass
class ProfilingExtension(SamplingExtension):
    """
    Extension to log memory usage before and after each step in the search process.
    """
    start_mem: int = None

    def __post_init__(self):
        if psutil is None:
            raise ImportError('psutil is required for ProfilingExtension')

    def before_step(self, step: str, scenario: Scenario, tod: TOD) -> None:
        super().before_step(step, scenario, tod)
        if self.should_sample:
            self.start_mem = psutil.Process().memory_info().rss

    def after_step(self, step: str, scenario: Scenario, tod: TOD) -> None:
        if self.should_sample and self.start_mem is not None:
            end_mem = psutil.Process().memory_info().rss
            memory_diff = end_mem - self.start_mem
            log.debug(f'[EXTENSION PROFILING] Step {step} increased memory by {memory_diff / 1024**2:.2f} MB')
        super().after_step(step, scenario, tod)

@dataclass
class TimingExtension(SamplingExtension):
    """
    Extension to log the time taken by each step in the search process, sampling approximately some % of the calls.
    """
    start_time: float = None

    def before_step(self, step: str, scenario: Scenario, tod: TOD) -> None:
        super().before_step(step, scenario, tod)
        if self.should_sample:
            self.start_time = time.time()

    def after_step(self, step: str, scenario: Scenario, tod: TOD) -> None:
        if self.should_sample and self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            log.debug(f'[EXTENSION TIMING] Step {step} took {elapsed_time:.2f} seconds')
        super().after_step(step, scenario, tod)

@dataclass
class PlottingExtension(SamplingExtension):
    """
    Extension to plot the data before and/or after each step in the search process.
    By default, it will plot after each step.
    """
    before: bool = False
    after: bool = True
    output_path: str = '.'
    count: int = 0
    format: str = 'png'
    dpi: int = 300

    def __post_init__(self):
        os.makedirs(self.output_path, exist_ok=True)

    def before_step(self, step: str, scenario: Scenario, tod: TOD) -> None:
        super().before_step(step, scenario, tod)
        if self.should_sample and self.before:
            det = self.detector_index(tod)
            self.plot(step, scenario, tod, det, 'before')

    def after_step(self, step: str, scenario: Scenario, tod: TOD) -> None:
        if self.should_sample and self.after:
            det = self.detector_index(tod)
            self.plot(step, scenario, tod, det, 'after')
        super().after_step(step, scenario, tod)

    # TODO: have this plot be part of ui/plot
    def plot(self, step: str, scenario: Scenario, tod: TOD, det: int, when: str) -> None:

        data = tod.data[det]
        
        mean = np.mean(data)
        max = np.argmax(data)

        _, ax1 = plt.subplots(1,1, figsize=(20, 8))

        ax1.plot(data)
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('Flux [mJy]')

        ax1.scatter(max, data[max], s=50, edgecolors='k', label=f'Max: {data[max]:.2f} mJy')
        ax1.axhline(y=mean, linestyle='--', linewidth=1, label=f'Mean: {mean:.2f} mJy')

        # Calculate the largest power of 10 ticks that fits into the range
        num_ticks = 5
        min, max = 0, len(data)
        diff = max - min
        max_power_of_10 = 10 ** int(np.log10(diff / num_ticks))
        tick_interval = int(np.ceil(diff / num_ticks / max_power_of_10) * max_power_of_10)
        sample_ticks = np.arange(min, max + tick_interval, tick_interval)
        ax1.set_xticks(sample_ticks)

        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(sample_ticks)

        # If data range is less than 2 minutes, show time in seconds
        if diff / tod.sampling_rate < 2*60:
            ax2.set_xlabel('Time (sec)')
            time_labels = sample_ticks / tod.sampling_rate
        else:
            ax2.set_xlabel('Time (min)')
            time_labels = sample_ticks / (tod.sampling_rate * 60)

        ax2.set_xticklabels(["{:.2f}".format(t) for t in time_labels])
        
        ax1.legend()

        ax1.grid(True)
        ax2.grid(False)

        title = [
            f'Detector {det}',
            f'TOD {tod.id}',
            f'{when} step {step}',
        ]

        seed = ''
        if scenario is not None:
            title.extend([
                f'Scenario {scenario.title}',
                f'Seed {scenario.seed}',
            ])
            seed = f'{scenario.seed}_'

        plt.title(' - '.join(title))
        self.count += 1

        filename = f'{tod.id}_{det}_{seed}{self.count}_{when}_{step}.{self.format}'

        plt.savefig(os.path.join(self.output_path, filename), format=self.format, dpi=self.dpi)
        log.debug(f'[EXTENSION PLOTTING] Plot saved to {filename}')
        plt.show()