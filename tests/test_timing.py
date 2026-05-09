import os
from unittest.mock import patch

import numpy as np
import pytest

from pulsar import Target
from pulsar import timing as timing_module
from pulsar.timing import BarycentricTimingModel, PeriodTimingModel


@pytest.fixture
def target():
    return Target(name='t', ra=0.5, dec=0.3, T=2.0, t0=10.0, phi0=0.25, phase_sign='-')


@pytest.fixture
def solution_file(tmpdir):
    path = os.path.join(str(tmpdir), 'sol.txt')
    # Columns 3, 4, 6, 8 are used: (tref_mjd, t0, freq_Hz, nudot * 1e-15)
    with open(path, 'w') as f:
        f.write("# header\n")
        f.write("0 0 0 50000.0 0.0 0 30.0 0 -1.0 0 0\n")
        f.write("0 0 0 50100.0 1.0 0 31.0 0 -2.0 0 0\n")
    return path


@pytest.fixture
def model(target, solution_file):
    """Barycentric model constructed from the test fixtures, with no ephem."""
    return BarycentricTimingModel(target=target, solution_file=solution_file)


class TestPeriodTimingModel:
    def test_obstime2phase_negative_sign(self, target):
        target.phase_sign = '-'
        m = PeriodTimingModel(target)
        # phi = (phi0 - (ctime - t0)/T) % 1
        # ctime = 12, t0 = 10, T = 2 -> phi = (0.25 - 1.0) % 1 = 0.25
        assert m.obstime2phase(12.0) == pytest.approx(0.25)

    def test_obstime2phase_positive_sign(self, target):
        target.phase_sign = '+'
        m = PeriodTimingModel(target)
        # phi = (0.25 + 1.0) % 1 = 0.25
        assert m.obstime2phase(12.0) == pytest.approx(0.25)

    def test_obstime2phase_wraps_modulo_one(self, target):
        target.phase_sign = '+'
        m = PeriodTimingModel(target)
        # ctime far enough to make the unwrapped value > 1
        # phi = (0.25 + 100/2) % 1 = (0.25 + 50.0) % 1 = 0.25
        assert m.obstime2phase(210.0) == pytest.approx(0.25)


class TestResolveEphem:
    """`_resolve_ephem` runs from `__init__`, so each case constructs a real
    BarycentricTimingModel with `urlretrieve` patched and inspects the resolved
    `ephem` attribute."""

    def _build(self, target, solution_file, ephem):
        with patch.object(timing_module, 'urlretrieve') as mock:
            m = BarycentricTimingModel(target=target, solution_file=solution_file, ephem=ephem)
        return m, mock

    def test_url_ephem_is_downloaded(self, target, solution_file):
        m, mock = self._build(target, solution_file, 'https://x/de.bsp')
        mock.assert_called_once()
        assert m.ephem.endswith('.bsp')
        assert not m.ephem.startswith('http')

    def test_local_path_ephem_is_passed_through(self, target, solution_file):
        m, mock = self._build(target, solution_file, '/tmp/local.bsp')
        mock.assert_not_called()
        assert m.ephem == '/tmp/local.bsp'

    def test_none_ephem(self, target, solution_file):
        m, mock = self._build(target, solution_file, None)
        mock.assert_not_called()
        assert m.ephem is None


class TestLoadTimingData:
    def test_columns_parsed(self, model):
        # Two rows -> arrays of length 2
        assert model.tref.shape == (2,)
        assert model.t0.shape == (2,)
        np.testing.assert_allclose(model.freq, [30.0, 31.0])
        np.testing.assert_allclose(model.dfreq, [-1.0e-15, -2.0e-15])
        np.testing.assert_allclose(model.P, [1 / 30.0, 1 / 31.0])
        np.testing.assert_allclose(
            model.dP, [-model.dfreq[0] / 30.0 ** 2, -model.dfreq[1] / 31.0 ** 2]
        )


class TestArithmeticHelpers:
    def test_calc_x(self, model):
        # log1p(0)/dP = 0 — verifies the formula at t=0
        assert model._calc_x(t=0.0, P=1.0, dP=1e-3) == pytest.approx(0.0)
        # Non-trivial: log1p(dP/P * t) / dP for small t
        t, P, dP = 1.0, 0.1, 1e-3
        expected = np.log1p(dP / P * t) / dP
        assert model._calc_x(t, P, dP) == pytest.approx(expected)

    def test_tai2tdt_adds_32_184(self, model):
        assert model._tai2tdt(0.0) == pytest.approx(32.184)
        assert model._tai2tdt(100.0) == pytest.approx(132.184)

    def test_ctime2jd_returns_finite(self, model):
        jd = model._ctime2jd(0.0)
        assert np.isfinite(jd)
        # Unix epoch in JD ≈ 2440587.5
        assert jd == pytest.approx(2440587.5, abs=1.0)

    @pytest.mark.parametrize('ctime', [0.0, 1e8, 1e9, 1.7e9])
    def test_calc_tdb_off_bounded(self, model, ctime):
        # Sinusoidal correction with amplitude 0.001658 -> bounded.
        assert abs(model._calc_tdb_off(ctime)) <= 0.002

    def test_calc_ind_x_returns_index_within_tref(self, model):
        # Pick a ctime just after the first tref entry — searchsorted side='right' minus 1 = 0.
        ind, x = model._calc_ind_x(model.tref[0] + 1.0)
        assert ind == 0
        assert np.isfinite(x)
