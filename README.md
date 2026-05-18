# Pulsar Signal Search Library
![Coverage](../../raw/badges/tests/coverage.svg)
<!-- Add Zenodo DOI badge below this line once v1.0.0 is archived (see Citation section):
[![DOI](https://zenodo.org/badge/DOI/<ZENODO_DOI>.svg)](https://doi.org/<ZENODO_DOI>)
-->

> **Citation:** If you use `pulsar-search` in your work, please cite both the
> software and the companion paper (arXiv:[2509.11960](https://doi.org/10.48550/arXiv.2509.11960),
> accepted by *The Astrophysical Journal*). See [`CITATION.cff`](./CITATION.cff)
> for machine-readable metadata.

## Introduction

`pulsar-search` is a Python package designed for simulating, filtering, and analysing pulsar signals in astronomical data. Its main usage is to estimate the flux amplitude of repeating signals in time-domain data by performing a targeted template search. Signal simulations can be injected into real or mock data to validate the pipeline and estimate instrument sensitivity.

The package was originally designed to use time-ordered data from the Atacama Cosmology Telescope (ACT), but it is structured to accommodate additional telescopes by implementing the [`instrument`](./pulsar/instrument.py) protocol.

An instrument should be able to provide time-ordered data as numpy arrays, map sky coordinates into time domain, and inject/retrieve signal amplitude. See the [`MockInstrument`](./pulsar/mock_instrument/instrument.py) for an example on how a new data source can be implemented.

## 🔧 Installation

This package isn't on PyPI. To install it:

1. Clone the repository:
   ```bash
   git clone https://github.com/louisbranch/pulsar-search.git
   cd pulsar-search
   ```

2. Install the base package:
   ```bash
   pip install .
   ```

---

### 🧩 Optional features

You can enable extra functionality by specifying one or more "extras" during install:

- **Profiling tools** (`psutil`):
  ```bash
  pip install .[profiling]
  ```

- **Parallel processing** (via `mpi4py`):
  ```bash
  pip install .[parallel]
  ```

- **ACT data support** (installs from GitHub):
  ```bash
  pip install .[act]
  ```

---

### 🛠 Dev environment

For development or testing:
```bash
pip install -e .[test]
```

Then run the suite with `make test` or `make coverage`.

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for the full development workflow,
extension points (`Instrument`, `TimingModel`, `SignalProfile`), and how to
file issues.

### Try it without telescope data

The `examples/crab_search_mock.py` script exercises the basic pipeline
(instrument wiring, TOD generation, filtering, profile construction) against
the bundled `MockInstrument`, so you can confirm a fresh install works
end-to-end without needing real data:

```bash
python examples/crab_search_mock.py
```

Full flux estimation requires a real `PointingModel`; see the ACT
implementation under `pulsar/act/` for a worked example.

## Configuration

The `pulsar-search` package allows customization of the instrument used and the verbosity of the logging system.

To adjust the instrument or logging level, use the optional `init` function:

```python
import logging

import pulsar
from your_instrument_module import YourInstrumentClass

pulsar.init(instrument=YourInstrumentClass(), log_level=logging.INFO)
```

Logging is set to `WARNING` by default.

## Basic Usage

### Reading Time-ordered data (TOD)

An instrument implements reading data as a generator, yielding one file at time. For example, to read and calibrate the first 10 files, selecting a subset of detectors:

```python
for tod in tods('input_path', limit=10, dets=[0, 2, 4]):
    tod.calibrate()
```

For parallel processing, it is recommended first to get a list of suitable ids with `tod_ids('input_path')` and then passing the desired ones to `tods('input_path', ids=ids)` after splitting the list by process.

### Filters

The library has three types of highpass filters:

* **Butterworth**: requires a cutoff frequency and an order controlling roll-off steepness.
* **FFT-based**: requires a knee frequency and a slope parameter controlling attenuation rate.
* **Planet**: excludes a sky region (RA, Dec, radius) and applies an FFT-based filter.

For example, applying the FFT-based filter on a previously loaded TOD:

```python
highpass_filter_fft(tod, fknee=3, alpha=10)
```

### Targets

A target represents an astronomical source with position, size, and timing attributes. A target is required for both simulations and flux estimation.

For example, to represent the Crab pulsar (`Target` takes coordinates and radius in radians):

```python
import numpy as np

target = Target(
    name='crab',
    ra=np.deg2rad(83.63),
    dec=np.deg2rad(22.01),
    radius=np.deg2rad(0.1),
    T=0.0335,
    D=0.1,
)
```

When loading from a TOML file via `TargetContext`, the loader handles the degree → radian conversion for you.

### Simulation Profiles

The library includes three profiles that can be injected into tod:

* **Boxcar**: models a top-hat shape, constructed by splitting the signal into discrete phase bins, one of which is "on".
* **von Mises**: a more realistic pulse shape using a gaussian-like circular profile. It uses the target's rotational period and duty cycle to build each pulse.
* **Constant**: simulates a constant source, for example a background nebula, to analyse its effects near a pulsar.

By default, the repeating signals are generated assuming a constant period and phase. However, the library also includes a `BarycentricTimingModel` that can offer higher precision by using a timing solution and solar system ephemeris. Additional models can be implemented through the `TimingModel` protocol.

For example, to build a boxcar profile with 10 discrete bins, active on the first bin:

```python
boxcar = Boxcar(num_bins=10, bin_index=0)
```

The library also has two helpers that allow the creation of multiples profiles at once. See `create_boxcar_profiles` and `create_von_mises_profiles`.

Custom profiles can be implemented via the `SignalProfile` abstract class.

## Searching

A search encapsulates the process of going through multiple TOD files calibrating, filtering, optionally injecting simulations, and finally estimating the flux amplitude of a target. A search can run in parallel by using `mpi4py`. It also supports different scenarios at once, each resulting in a separated search result file (`.hdf5`) containing the right-hand side and normal matrix values for flux estimation.

### Target Context

Since a search usually involves the same target, timing, and filter options, a context can be defined as a `.toml` file. For example:

If `crab.toml` is defined as:

```toml
[target]
name = "crab"
ra = 83.63322         # Right Ascension in degrees
dec = 22.01446        # Declination in degrees
amp = 0               # Expected amplitude in mJy (0 if unknown)
T = 0.0335            # Rotational period in seconds
phi0 = 0.0            # Initial phase
D = 0.1               # Duty cycle (fraction of period)
R = 0.2               # Radius in degrees

[timing]
solution_file = "data/act/timing/crab.txt" 
ephem = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de200.bsp"  # Ephemeris file URL (will be downloaded if not local)

[filter]
method = 'planet'
fknee = 3
alpha = 10
```

Then it can be loaded with:

```python
ctx = TargetContext('crab.toml')
crab = ctx.target
filter = ctx.filter
timing = ctx.timing_model
```

### Defining a Scenario

A search scenario represents a set of target profiles and operations to be applied to a list of TOD.
Each scenario is uniquely identified by a seed, which is generated based on the scenario parameters.

Before the search is initiated, each scenario checks which ids are missing and skips the ones already processed. Therefore, one can perform incremental searches by increasing the `tod_limit` or by setting different `tod_ids` each time the search runs.

#### Example: searching the Crab pulsar using a boxcar profile
        
```python
ctx = TargetContext('crab.toml')
target = ctx.target

search_profiles = create_boxcar_profiles(10, target)

scenario1 = Scenario(
    title='Search for the Crab pulsar',
    target=target,
    search_profiles=search_profiles,
    filter=ctx.filter,
)
```

#### Example: injecting a simulation and add noise before searching

```python
ctx = TargetContext('crab.toml')
target = ctx.target

noise = MultiplicativeGaussianNoise(mean=1.0, std=0.1)
search_profiles = create_boxcar_profiles(10, target)
simulation = SignalOperation(profile=VonMisesProfile(target), amp=10_000)

scenario2 = Scenario(
    title='Search for the Crab pulsar simulation',
    target=target,
    search_profiles=search_profiles,
    filter=ctx.filter,
    noise=noise,
    operations=[simulation],
    metadata={
        'comments': 'Add a 10 Jy simulation von Mises profile'
    }
)
```

#### Example: applying a planet filter to the background nebula and search for an offset target

```python
import astropy.units as u

ctx = TargetContext('crab.toml')
target = ctx.target
off_target = target.offset_position(distance=30.0, direction='east', unit=u.arcmin)
off_target.name = 'off-target by 30 arcmin east'

noise = MultiplicativeGaussianNoise(mean=1.0, std=0.1)
search_profiles = create_boxcar_profiles(10, off_target)
simulation = SignalOperation(profile=VonMisesProfile(off_target), amp=10_000)

scenario3 = Scenario(
    title='Search for an off-target Crab pulsar simulation',
    target=off_target,
    search_profiles=search_profiles,
    filter=ctx.filter,
    noise=noise,
    operations=[simulation],
    filter_sources=[target],
    metadata={
        'comments': 'Add a 10 Jy simulation von Mises profile to an offset location'
    }
)
```

### Running the Search

Running the scenarios in the examples above against the first 10 TOD files:

```python
search = Search(
    scenarios=[scenario1, scenario2, scenario3],
    tod_path='input_path',
    output_path='output_path',
    tod_limit=10,
)

search.run()
```

If MPI is available, the search will run in parallel, otherwise it is performed sequentially.

### Reading the Results

Each scenario will generate a `.hdf5` result that can be read by the `Storage.read` method:

```python
data, metadata, references = Storage.read('output_path/seed_id.hdf5')
```

The data is a numpy array with the TOD id, the right-hand side, and the normal matrix values.
Metadata contains additional information about the search, such as title, start and end times, and comments.
References contain the TOD id, search profile index, and the shape of the right-hand side and normal matrix arrays.

### Estimating the Flux

Finally, with the data results for a scenario we can estimate the flux:

```python
estimator = pulsar.FluxEstimator(pmat=config.instrument.pointing_model)

raw_fluxes, fluxes, mean_flux, err, snr = estimator.calculate_flux(data, nsplits=20, subtract_median_flux=True)
```

Both `raw_fluxes` and `fluxes` have the shape (nsplits, n_profiles, n_components). Using the examples above, it would be `(20, 10, 3)` since we created 10 boxcar profiles and we are using the default number of polarization components (T, Q, U). While `mean_flux`, the standard error, and SNR are calculated across the splits, in this case `(10, 3)`.