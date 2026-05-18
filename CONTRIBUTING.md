# Contributing

Thanks for your interest in `pulsar-search`. This document covers how to set
up a development environment, run the test suite, file issues, and extend the
package against new data sources.

## Development setup

Clone the repo and install the package in editable mode with the `test` extra:

```bash
git clone https://github.com/louisbranch/pulsar-search.git
cd pulsar-search
pip install -e .[test]
```

This pulls in the core scientific Python stack (`numpy`, `scipy`, `astropy`,
`pixell`, `h5py`, `matplotlib`, …) plus `pytest`, `pytest-cov`, and
`genbadge[coverage]`.

The `[parallel]` extra (`mpi4py`) is only needed if you want to run
`Search.run_parallel()`; sequential runs and the test suite do not require it.
The `[act]` extra installs `enlib` and `enact` from GitHub and is only needed
to use the bundled ACT instrument or `BarycentricTimingModel`.

## Running the tests

```bash
make test          # pytest -q
make coverage      # pytest with coverage XML output
```

The suite mocks `enlib`, `enact`, and `mpi4py` via `tests/conftest.py` (see
`tests/_act_stubs.py`), so a clean dev environment without those extras runs
the full suite.

There is also a smoke-test script that exercises the basic pipeline against
the bundled `MockInstrument` without any telescope data:

```bash
python examples/crab_search_mock.py
```

## Filing issues

Open an issue at https://github.com/louisbranch/pulsar-search/issues. Useful
context to include: Python version, install command used, full traceback, and
a minimal snippet that reproduces the problem.

## Extending the package

The package is organised around three protocols. Implementing any of them is
how you teach `pulsar-search` about a new telescope, timing model, or pulse
shape.

- **`pulsar.Instrument`** (`pulsar/instrument.py`) — yields time-ordered data
  and provides a pointing model. See `pulsar.mock_instrument.MockInstrument`
  for the smallest reference implementation, and `pulsar/act/` for the
  production ACT implementation.
- **`pulsar.TimingModel`** (`pulsar/timing.py`) — converts observation time to
  pulsar phase. `PeriodTimingModel` is the constant-period reference;
  `BarycentricTimingModel` adds full timing corrections (requires `[act]`).
- **`pulsar.SignalProfile`** (`pulsar/signal_profile.py`) — describes a pulse
  shape across phase. `BoxcarProfile`, `VonMisesProfile`, and
  `ConstantProfile` are the bundled implementations.

When adding a new implementation, prefer composing the existing protocols
over modifying core modules. Add tests under `tests/` mirroring the package
structure.

## Code style

The CI runs `flake8` with E9/F63/F7/F82 as hard errors and a 127-column line
limit. There is no enforced formatter; match the surrounding style.

## License

By contributing you agree that your contributions are licensed under the
project's BSD 3-Clause license. See `LICENSE` for the full text and
`AUTHORS.md` for the contributor list.
