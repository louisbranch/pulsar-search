from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pulsar.act.map import create_map, plot_map


def _make_tod(ndet=2, nsamp=16, srate=400.0):
    tod = MagicMock(name='TOD')
    tod.data = np.zeros((ndet, nsamp))
    tod.num_samples = nsamp
    tod.scan = MagicMock(name='Scan')
    tod.scan.srate = srate
    return tod


class TestCreateMap:
    def test_returns_three_channel_map_with_expected_shape(self):
        tod = _make_tod()
        omap = create_map(tod, coord=(0.5, -0.3), geometry=(0.01, (4, 5)))
        # The default Enmap stub returns shape == pixels.
        assert omap.shape == (3, 4, 5)
        assert omap.dtype == np.float32

    def test_geometry_called_with_dec_ra_swap(self):
        tod = _make_tod()
        from enlib import enmap
        with patch.object(enmap, 'geometry', wraps=enmap.geometry) as mock_geom:
            create_map(tod, coord=(1.0, 2.0), geometry=(0.005, (3, 3)))
        # First arg to geometry is (dec, ra) — note the swap from coord (ra, dec).
        called_coord = mock_geom.call_args.args[0]
        assert called_coord == (2.0, 1.0)


class TestPlotMap:
    def test_calls_pshow_on_selected_component(self):
        from pixell import enplot
        omap = np.zeros((3, 4, 5))
        with patch.object(enplot, 'pshow') as mock_pshow:
            plot_map(omap, component=2)
        mock_pshow.assert_called_once()
        passed = mock_pshow.call_args.args[0]
        np.testing.assert_array_equal(passed, omap[2])

    def test_default_component_is_zero(self):
        from pixell import enplot
        omap = np.zeros((3, 4, 5))
        with patch.object(enplot, 'pshow') as mock_pshow:
            plot_map(omap)
        passed = mock_pshow.call_args.args[0]
        np.testing.assert_array_equal(passed, omap[0])
