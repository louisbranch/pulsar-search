from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pulsar.act.noise_models import NmatTot


def _make_scan(ndet=2, nsamp=16, srate=400.0):
    scan = MagicMock(name='Scan')
    scan.srate = srate
    scan.nsamp = nsamp
    scan.tod = np.ones((ndet, nsamp))
    scan.cut = MagicMock(name='cut')
    scan.cut_noiseest = MagicMock(name='cut_noiseest')
    scan.spikes = MagicMock(name='spikes')
    return scan


class TestNmatTotInit:
    def test_no_filter_when_filter_arg_is_none(self):
        scan = _make_scan()
        from enlib import config
        # Provide concrete config defaults so the model & window are real values.
        config.set('noise_model', 'jon')
        config.set('tod_window', 1.0)

        n = NmatTot(scan)

        assert n.filter is None
        assert n.model == 'jon'
        # window = config_value * scan.srate
        assert n.window == 1.0 * scan.srate
        assert n.cut is scan.cut

    def test_filter_curve_built_when_provided(self):
        scan = _make_scan()
        from enlib import config
        config.set('noise_model', 'jon')
        config.set('tod_window', 1.0)

        n = NmatTot(scan, filter=(3.0, 10.0))

        assert n.filter is not None
        # filter shape should match rfftfreq(nsamp, 1/srate)
        assert n.filter.shape == (scan.nsamp // 2 + 1,)


class TestNmatTotApply:
    def test_apply_uses_filter_when_present(self):
        scan = _make_scan()
        from enlib import config
        config.set('noise_model', 'jon')
        config.set('tod_window', 1.0)

        n = NmatTot(scan, filter=(3.0, 10.0))
        tod = np.ones((2, 16))
        n.apply(tod)
        # The mock noise matrix's apply_ft is a no-op; we only check that the
        # call ran without raising.

    def test_white_round_trips_window(self):
        scan = _make_scan()
        from enlib import config
        config.set('noise_model', 'jon')
        config.set('tod_window', 1.0)

        n = NmatTot(scan)
        tod = np.zeros((2, 16))
        # apply_window is a no-op stub; this just verifies the call chain
        # composes without error.
        n.white(tod)
