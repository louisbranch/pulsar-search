from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pulsar.act.pointing_models import PmatTotTransient


def _make_scan(ndet=2, nsamp=8):
    scan = MagicMock(name='ACTScan')
    scan.ndet = ndet
    scan.nsamp = nsamp
    scan.mjd0 = 50000.0
    scan.boresight = np.zeros((3, nsamp))
    scan.cut = MagicMock(name='cut')
    scan.d = MagicMock(name='scan.d')
    scan.d.point_correction = np.zeros(2)
    scan.d.point_template = np.zeros(2)
    return scan


def _make_tod(scan):
    tod = MagicMock(name='TOD')
    tod.scan = scan
    return tod


def _make_source(ra, dec, amp=1.0):
    src = MagicMock(name='Source')
    src.src = np.array([ra, dec, amp])
    src.obstime2profile = lambda ctime: np.zeros_like(ctime)
    return src


class TestInit:
    def test_stores_ncomp_and_params_shape(self):
        scan = _make_scan()
        tod = _make_tod(scan)
        srcs = [_make_source(1.0, 0.5), _make_source(1.5, 0.6)]

        p = PmatTotTransient(tod, srcs, ncomp=3)

        assert p.ncomp == 3
        # params shape: [nsrcs, ndir=1, ndet|1, 8]
        assert p.params.shape == (2, 1, 1, 8)
        # RA/Dec swapped into the first two columns of `params[:, :, :, :2]`.
        np.testing.assert_allclose(p.params[0, 0, 0, 0], 0.5)  # dec
        np.testing.assert_allclose(p.params[0, 0, 0, 1], 1.0)  # ra
        # Beam size goes into columns 5:7 (default (1, 1)).
        np.testing.assert_allclose(p.params[0, 0, 0, 5:7], [1.0, 1.0])

    def test_perdet_expands_third_axis(self):
        scan = _make_scan(ndet=4)
        tod = _make_tod(scan)
        srcs = [_make_source(1.0, 0.5)]

        p = PmatTotTransient(tod, srcs, perdet=True)

        assert p.params.shape == (1, 1, 4, 8)

    def test_custom_beam_size(self):
        scan = _make_scan()
        tod = _make_tod(scan)
        srcs = [_make_source(1.0, 0.5)]

        p = PmatTotTransient(tod, srcs, beam_size=(2.5, 3.5))

        np.testing.assert_allclose(p.params[0, 0, 0, 5:7], [2.5, 3.5])


class TestForward:
    def test_calls_psrc_forward_with_amps_in_correct_slot(self):
        scan = _make_scan()
        tod = _make_tod(scan)
        srcs = [_make_source(1.0, 0.5)]
        p = PmatTotTransient(tod, srcs, ncomp=3)

        with patch.object(p.psrc, 'forward') as mock_forward:
            data = np.zeros((2, 8))
            amps = np.array([[[[2.0, 3.0, 4.0]]]])  # shape (1, 1, 1, 3)
            p.forward(data, amps, pmul=1.5)

        mock_forward.assert_called_once()
        called_data, called_params = mock_forward.call_args.args
        assert called_data is data
        # Forward writes amps into params[..., 2:5] before calling psrc.forward.
        np.testing.assert_allclose(called_params[..., 2:5], amps)
        assert mock_forward.call_args.kwargs.get('pmul') == 1.5


class TestBackward:
    def test_returns_amplitude_slice_when_amps_is_none(self):
        scan = _make_scan()
        tod = _make_tod(scan)
        srcs = [_make_source(1.0, 0.5)]
        p = PmatTotTransient(tod, srcs, ncomp=3)

        # Pre-load known values in the amplitude slot of the underlying params.
        # Backward will copy from p.params, mutate via psrc.backward, then slice [..., 2:5].
        def fake_backward(data, params, pmul=1):
            params[..., 2:5] = np.array([5.0, 6.0, 7.0])

        with patch.object(p.psrc, 'backward', side_effect=fake_backward):
            amps = p.backward(np.zeros((2, 8)))

        np.testing.assert_allclose(amps[0, 0, 0], [5.0, 6.0, 7.0])

    def test_writes_into_provided_amps_buffer(self):
        scan = _make_scan()
        tod = _make_tod(scan)
        srcs = [_make_source(1.0, 0.5)]
        p = PmatTotTransient(tod, srcs, ncomp=3)

        def fake_backward(data, params, pmul=1):
            params[..., 2:5] = np.array([1.0, 2.0, 3.0])

        with patch.object(p.psrc, 'backward', side_effect=fake_backward):
            buffer = np.zeros((1, 1, 1, 3))
            out = p.backward(np.zeros((2, 8)), amps=buffer)

        assert out is buffer
        np.testing.assert_allclose(buffer[0, 0, 0], [1.0, 2.0, 3.0])


class TestSetOffset:
    def test_set_offset_updates_psrc_scan_offsets(self):
        scan = _make_scan()
        tod = _make_tod(scan)
        srcs = [_make_source(1.0, 0.5)]
        p = PmatTotTransient(tod, srcs)

        # set_offset writes through psrc.scan.offsets[:, 1:]; provide a real array.
        p.psrc.scan.offsets = np.zeros((3, 3))
        from enact import actdata
        with patch.object(actdata, 'offset_to_dazel', return_value=np.array([[1.0, 2.0]] * 3)) as mock_off:
            p.set_offset(np.array([0.1, 0.2]))
        mock_off.assert_called_once()
        np.testing.assert_allclose(p.psrc.scan.offsets[:, 1:], [[1.0, 2.0]] * 3)
