import os
from unittest.mock import patch

import numpy as np
import pytest

from pulsar.target_context import TargetContext
from pulsar.timing import BarycentricTimingModel, PeriodTimingModel


_BASE_TARGET_BLOCK = """
[target]
name = "crab"
ra = 83.63
dec = 22.01
amp = 10.0
T = 0.0335
phi0 = 0.0
D = 0.1
R = 0.2
"""


@pytest.fixture
def write_config(tmpdir):
    """Returns a writer that emits a TOML config file (target block + extras)
    and returns its path."""
    def write(extra_blocks: str = '') -> str:
        path = os.path.join(str(tmpdir), 'crab.toml')
        with open(path, 'w') as f:
            f.write(_BASE_TARGET_BLOCK + extra_blocks)
        return path
    return write


def test_target_fields_converted_to_radians(write_config):
    ctx = TargetContext(write_config())

    assert ctx.target.name == 'crab'
    assert ctx.target.ra == pytest.approx(np.deg2rad(83.63))
    assert ctx.target.dec == pytest.approx(np.deg2rad(22.01))
    assert ctx.target.radius == pytest.approx(np.deg2rad(0.2))


def test_period_timing_model_when_timing_block_absent(write_config):
    ctx = TargetContext(write_config())
    assert isinstance(ctx.timing_model, PeriodTimingModel)


def test_barycentric_timing_model_when_timing_block_present(tmpdir, write_config):
    solution_file = os.path.join(str(tmpdir), 'sol.txt')
    # Match the column layout expected by BarycentricTimingModel._load_timing_data.
    # Columns indexed (3, 4, 6, 8) -> (tref_mjd, t0, freq, nudot*1e-15)
    with open(solution_file, 'w') as f:
        f.write("# header\n")
        f.write("0 0 0 50000.0 0.0 0 30.0 0 -1.0 0 0\n")

    path = write_config("""
[timing]
solution_file = "sol.txt"
ephem = "https://example.com/de200.bsp"
""")

    with patch('pulsar.timing.urlretrieve') as mock_urlretrieve:
        ctx = TargetContext(path, root_dir=str(tmpdir) + '/')

    mock_urlretrieve.assert_called_once()
    assert isinstance(ctx.timing_model, BarycentricTimingModel)


def test_filter_defaults_when_block_absent(write_config):
    ctx = TargetContext(write_config())

    assert ctx.filter.method == 'butterworth'
    assert ctx.filter.cutoff == 0
    assert ctx.filter.fknee == 0
    assert ctx.filter.alpha == 0


def test_filter_block_populates_options(write_config):
    path = write_config("""
[filter]
method = 'fft'
fknee = 3
alpha = 10
""")
    ctx = TargetContext(path)

    assert ctx.filter.method == 'fft'
    assert ctx.filter.fknee == 3
    assert ctx.filter.alpha == 10
