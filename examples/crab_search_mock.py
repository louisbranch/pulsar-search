"""End-to-end smoke test using `MockInstrument`.

Runs the parts of the pulsar-search pipeline that don't require telescope
data, so a fresh install can be exercised without ACT credentials or any
real time-ordered data. Prints a short summary; non-zero exit on failure.

Usage:
    python examples/crab_search_mock.py

What this covers:
- `pulsar.init(MockInstrument())` — wiring a non-ACT instrument into the
  global config.
- TOD generation via `pulsar.tods()`.
- A Butterworth high-pass filter, exercising `pulsar.filters` (and
  transitively scipy/numpy).
- Construction of a `Target` (Crab pulsar coordinates) and a set of
  `BoxcarProfile`s via `create_boxcar_profiles`.
- A `PeriodTimingModel` evaluated on a sample timestamp.

What this does *not* cover:
- Flux estimation. `FluxEstimator` requires a real `PointingModel`;
  `MockInstrument.pointing_model()` returns None by design. A working
  end-to-end search requires an instrument implementation backed by real
  pointing data (see `pulsar/act/` for the ACT-backed example).
"""
import logging
import sys

import numpy as np

import pulsar
from pulsar import (
    FilterOptions,
    PeriodTimingModel,
    Target,
    create_boxcar_profiles,
    filter,
    tods,
)
from pulsar.mock_instrument import MockInstrument


def main() -> int:
    pulsar.init(
        instrument=MockInstrument(
            max_tods=3,
            dets_per_tod=4,
            samples_per_tod=4_000,
            sampling_rate=400.0,
        ),
        log_level=logging.WARNING,
    )

    crab = Target(
        name="crab",
        ra=np.deg2rad(83.63322),
        dec=np.deg2rad(22.01446),
        radius=np.deg2rad(0.1),
        T=0.0335,
        D=0.1,
    )
    print(f"Target: {crab.name}  RA={crab.ra:.4f} rad  Dec={crab.dec:.4f} rad")

    butterworth = FilterOptions(method="butterworth", cutoff=1.0, order=2)

    n_tods = 0
    for tod in tods("ignored-by-mock"):
        before_std = float(np.std(tod.data))
        filter(tod, butterworth)
        after_std = float(np.std(tod.data))
        n_tods += 1
        print(
            f"  TOD {tod.id}: {tod.num_detectors} dets × {tod.num_samples} samples,"
            f" std {before_std:.3g} → {after_std:.3g} after high-pass"
        )
    assert n_tods > 0, "MockInstrument yielded no TODs"

    profiles = create_boxcar_profiles(num_bins=4, target=crab)
    assert len(profiles) == 4
    print(f"Built {len(profiles)} boxcar profiles (one per phase bin)")

    timing = PeriodTimingModel(crab)
    phase_at_t = timing.obstime2phase(0.5)
    print(f"PeriodTimingModel(crab).obstime2phase(0.5) = {phase_at_t:.6f}")

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
