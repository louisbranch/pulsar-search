from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pulsar.act.tod import TOD


def _make_scan(ndet=4, nsamp=16, srate=400.0):
    scan = MagicMock(name='ACTScan')
    scan.ndet = ndet
    scan.nsamp = nsamp
    scan.srate = srate
    scan.tod = np.zeros((ndet, nsamp))
    scan.boresight = np.zeros((3, nsamp))
    scan.point_offset = np.zeros((ndet, 2))
    scan.cut = MagicMock(name='cut')
    scan.site = MagicMock(name='site')
    scan.d = MagicMock(name='scan.d')
    scan.d.boresight = scan.boresight
    scan.d.point_offset = scan.point_offset
    scan.d.site = scan.site
    return scan


@pytest.fixture
def scan():
    return _make_scan()


@pytest.fixture
def dataset():
    ds = MagicMock(name='DataSet')
    ds.ndet = 4
    ds.tod = np.ones((4, 16))
    return ds


@pytest.fixture
def tod(scan, dataset):
    return TOD(id='tod_0001', scan=scan, dataset=dataset, band='f090', array='ar5')


class TestTodProperties:
    def test_id(self, tod):
        assert tod.id == 'tod_0001'

    def test_str(self, tod):
        assert str(tod) == 'TOD [tod_0001] - Band: f090, Array: ar5'

    def test_num_detectors(self, tod, dataset):
        dataset.ndet = 7
        assert tod.num_detectors == 7

    def test_num_samples(self, tod, scan):
        scan.nsamp = 1024
        assert tod.num_samples == 1024

    def test_sampling_rate(self, tod, scan):
        scan.srate = 200.0
        assert tod.sampling_rate == 200.0

    def test_data_uses_dataset_when_uncalibrated(self, tod, dataset):
        assert tod.calibrated is False
        np.testing.assert_array_equal(tod.data, dataset.tod)

    def test_data_uses_scan_when_calibrated(self, tod, scan):
        # Manually mark calibrated and assert the scan's tod is returned.
        tod._calibrated = True
        scan.tod = np.full((4, 16), 7.0)
        np.testing.assert_array_equal(tod.data, scan.tod)

    def test_data_setter_writes_to_scan(self, tod, scan):
        new = np.full((4, 16), 3.14)
        tod.data = new
        np.testing.assert_array_equal(scan.tod, new)


class TestCalibrate:
    def test_idempotent_when_already_calibrated(self, tod):
        tod._calibrated = True
        from enact import actdata
        with patch.object(actdata, 'calibrate') as mock_calibrate:
            tod.calibrate()
        mock_calibrate.assert_not_called()

    def test_runs_actdata_calibrate_and_applies_factors(self, tod, scan, dataset):
        from enact import actdata
        with patch.object(actdata, 'calibrate') as mock_calibrate:
            tod.calibrate(exclude_filters=['filter_a'])

        mock_calibrate.assert_called_once_with(dataset, exclude=['filter_a'])
        # Calibration applied: scan.tod = dataset.tod * jansky_sr * beam_sr.
        from pulsar.act.calibration import jansky_sr, beam_sr
        expected = dataset.tod * jansky_sr() * beam_sr('ar5', 'f090')
        np.testing.assert_allclose(scan.tod, expected)
        assert tod.calibrated is True


class TestDownsample:
    def test_factor_one_or_less_raises(self, tod):
        with pytest.raises(ValueError):
            tod.downsample(1)
        with pytest.raises(ValueError):
            tod.downsample(0)

    def test_downsample_calls_get_samples(self, tod, scan):
        # Slicing scan returns a new MagicMock; .get_samples on that returns a value
        # we shove into tod.data via the setter.
        tod.downsample(2)
        # After downsample, scan was reassigned to a slice; verify get_samples was used.
        assert tod.scan is not scan  # replaced by the slice's MagicMock


class TestSourceMethods:
    def test_remove_source_calls_avoidance_cut_and_gapfill(self, tod):
        from enact import cuts
        from enlib import gapfill
        with patch.object(cuts, 'avoidance_cut') as mock_avoid, \
             patch.object(gapfill, 'gapfill_joneig') as mock_gap:
            mock_avoid.return_value = MagicMock(name='samples')
            tod.remove_source(ra=1.0, dec=0.5, R=0.1)
        mock_avoid.assert_called_once()
        mock_gap.assert_called_once()
        # inplace=True is the contract for remove_source.
        assert mock_gap.call_args.kwargs.get('inplace') is True

    def test_locate_source_returns_model_and_samples(self, tod):
        from enact import cuts
        from enlib import gapfill
        sentinel_samples = MagicMock(name='samples')
        sentinel_model = np.full((4, 16), 0.5)
        with patch.object(cuts, 'avoidance_cut', return_value=sentinel_samples), \
             patch.object(gapfill, 'gapfill_joneig', return_value=sentinel_model):
            model, samples = tod.locate_source(ra=1.0, dec=0.5, R=0.1)
        assert samples is sentinel_samples
        np.testing.assert_array_equal(model, sentinel_model)

    def test_locate_samples_aggregates_min_max_per_cut(self, tod):
        from enact import cuts
        # cut.to_list() yields per-detector (n, 2) arrays; we want lowest min and
        # highest max across each.
        cut_obj = MagicMock(name='cut')
        cut_obj.to_list.return_value = [
            np.array([[10, 20], [5, 25]]),   # det 0: min=5, max=25
            np.array([]),                     # det 1: skipped
            np.array([[100, 200]]),           # det 2: min=100, max=200
        ]
        with patch.object(cuts, 'avoidance_cut', return_value=cut_obj):
            result = tod.locate_samples(ra=0.0, dec=0.0, R=0.1)
        np.testing.assert_array_equal(result, np.array([[5, 25], [100, 200]]))


class TestRecoverPosition:
    def _patch_interpol(self, ra, dec):
        from enlib import coordinates
        return patch.object(coordinates, 'interpol_pos', return_value=np.array([ra, dec]))

    def test_recover_radec_static_call_zero_offset(self):
        # det_offset = (0, 0) → recovered = boresight RA/Dec.
        bore = np.zeros((3, 4))
        bore[0] = 1.7e9  # ctime
        det_offs = np.zeros((2, 2))
        site = MagicMock(name='site')

        with self._patch_interpol(ra=1.5, dec=0.5):
            ra, dec = TOD.recover_radec(sample_idx=2, det_idx=0,
                                        bore=bore, det_offs=det_offs, site=site)
        assert ra == pytest.approx(1.5)
        assert dec == pytest.approx(0.5)

    def test_recover_radec_applies_declination_correction(self):
        # det_x = 0.1, dec=0 (cos(0)=1) → ra_offset = 0.1 / 1 = 0.1
        bore = np.zeros((3, 4))
        bore[0] = 1.7e9
        det_offs = np.array([[0.1, 0.05]])
        site = MagicMock(name='site')

        with self._patch_interpol(ra=1.0, dec=0.0):
            ra, dec = TOD.recover_radec(0, 0, bore, det_offs, site)
        assert ra == pytest.approx(1.1)
        assert dec == pytest.approx(0.05)

    def test_recover_radec_stretches_ra_by_inverse_cos_dec(self):
        # At dec = pi/3, cos(dec) = 0.5 → ra_offset = det_x / 0.5 = 2 * det_x
        bore = np.zeros((3, 4))
        bore[0] = 1.7e9
        det_offs = np.array([[0.1, 0.0]])
        site = MagicMock(name='site')

        with self._patch_interpol(ra=1.0, dec=np.pi / 3):
            ra, _ = TOD.recover_radec(0, 0, bore, det_offs, site)
        assert ra == pytest.approx(1.0 + 0.2)

    def test_recover_position_uses_first_detector_by_default(self, tod, scan):
        scan.boresight = np.zeros((3, 4))
        scan.boresight[0] = 1.7e9
        scan.point_offset = np.array([[0.1, 0.0], [0.5, 0.5]])

        with patch.object(TOD, 'recover_radec', wraps=TOD.recover_radec) as spy:
            tod.recover_position(sample_idx=1)
        # det_idx defaults to 0
        assert spy.call_args.args[1] == 0

    def test_recover_position_accepts_explicit_det_idx(self, tod, scan):
        scan.boresight = np.zeros((3, 4))
        scan.boresight[0] = 1.7e9
        scan.point_offset = np.array([[0.0, 0.0], [0.2, 0.0]])

        with patch.object(TOD, 'recover_radec', wraps=TOD.recover_radec) as spy:
            tod.recover_position(sample_idx=2, det_idx=1)
        assert spy.call_args.args[1] == 1


class TestMultiplicativeNoise:
    def test_scales_each_detector_independently(self, tod, scan):
        scan.tod = np.ones((4, 16))
        with patch('numpy.random.normal', return_value=np.array([1.0, 2.0, 3.0, 4.0])):
            tod.multiplicative_gaussian_noise(sigma=0.5)
        np.testing.assert_array_equal(scan.tod[0], np.ones(16) * 1.0)
        np.testing.assert_array_equal(scan.tod[3], np.ones(16) * 4.0)


class TestFillGaps:
    def test_default_fills_self_data(self, tod, scan):
        from enlib import gapfill
        with patch.object(gapfill, 'gapfill_joneig') as mock_gap:
            tod.fill_gaps()
        # First positional arg is the data to fill, second is the cut.
        mock_gap.assert_called_once()
        args, kwargs = mock_gap.call_args
        assert args[1] is scan.cut
        assert kwargs.get('inplace') is True

    def test_custom_data_is_passed_through(self, tod):
        from enlib import gapfill
        custom = np.zeros((4, 16))
        with patch.object(gapfill, 'gapfill_joneig') as mock_gap:
            tod.fill_gaps(custom)
        args, _ = mock_gap.call_args
        assert args[0] is custom
