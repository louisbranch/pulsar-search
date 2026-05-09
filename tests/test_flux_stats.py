import os
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest

from pulsar import Target
from pulsar.flux_stats import (
    FluxStat,
    FluxStatResult,
    FluxStats,
    compute_with_method,
    pad_with_nan,
)
from .test_mocks import create_mock_tod


def _stub_samples(mask_2d: np.ndarray):
    samples = MagicMock(name='Samples')
    samples.to_mask.return_value = mask_2d
    return samples


@pytest.fixture
def tod():
    t = create_mock_tod()
    t.data = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ])
    t.num_samples = 4
    t.num_detectors = 2
    return t


class TestCalculateFluxStatistics:
    def test_overall_only_when_target_is_none(self, tod):
        stats = FluxStats()._calculate_flux_statistics(tod, target=None)

        assert isinstance(stats, FluxStatResult)
        assert stats.target is None
        assert stats.background is None
        np.testing.assert_allclose(stats.overall.mean, [2.5, 6.5])
        np.testing.assert_allclose(stats.overall.median, [2.5, 6.5])
        np.testing.assert_allclose(stats.overall.min, [1.0, 5.0])
        np.testing.assert_allclose(stats.overall.max, [4.0, 8.0])

    def test_target_and_background_when_target_provided(self, tod):
        # mask=True means "in target region"; ~mask used to mask out non-target.
        in_target = np.array([
            [True, True, False, False],
            [True, True, False, False],
        ])
        tod.locate_source.side_effect = lambda ra, dec, R: (None, _stub_samples(in_target))

        target = Target(name='t', ra=0.5, dec=0.3, radius=0.1, T=1.0)
        stats = FluxStats()._calculate_flux_statistics(tod, target=target)

        assert stats.target is not None
        assert stats.background is not None
        # Target region: dets 0,1 use samples [0,1] -> means [1.5, 5.5]
        np.testing.assert_allclose(stats.target.mean, [1.5, 5.5])
        # Background region: samples [2,3] -> means [3.5, 7.5]
        np.testing.assert_allclose(stats.background.mean, [3.5, 7.5])


class TestSaveAndRead:
    def test_round_trip(self, tmpdir, tod):
        in_target = np.array([
            [True, True, False, False],
            [True, True, False, False],
        ])
        tod.locate_source.side_effect = lambda ra, dec, R: (None, _stub_samples(in_target))

        fs = FluxStats()
        target = Target(name='t', ra=0.5, dec=0.3, radius=0.1, T=1.0)
        stats = fs._calculate_flux_statistics(tod, target=target)

        path = os.path.join(str(tmpdir), 'stats.hdf5')
        fs._save_flux_statistics_to_hdf5(path, tod.id, stats)

        with h5py.File(path, 'r') as f:
            assert tod.id in f
            assert 'overall' in f[tod.id]
            assert 'target' in f[tod.id]
            assert 'background' in f[tod.id]
            np.testing.assert_allclose(f[tod.id]['overall']['mean'][:], stats.overall.mean)

        result = fs.read(path)
        assert result.overall is not None
        assert result.target is not None
        assert result.background is not None

    def test_read_skips_missing_target_and_background(self, tmpdir, tod):
        fs = FluxStats()
        stats = fs._calculate_flux_statistics(tod, target=None)
        path = os.path.join(str(tmpdir), 'stats.hdf5')
        fs._save_flux_statistics_to_hdf5(path, tod.id, stats)

        result = fs.read(path)
        assert result.target is None
        assert result.background is None


class TestCalculateAverageStatistics:
    def test_componentwise_mean(self):
        a = FluxStat(
            mean=np.array([1.0, 3.0]),
            median=np.array([2.0, 4.0]),
            std=np.array([0.1, 0.2]),
            min=np.array([0.0, 1.0]),
            max=np.array([2.0, 5.0]),
        )
        b = FluxStat(
            mean=np.array([5.0, 7.0]),
            median=np.array([6.0, 8.0]),
            std=np.array([0.3, 0.4]),
            min=np.array([4.0, 6.0]),
            max=np.array([8.0, 9.0]),
        )

        avg = FluxStats()._calculate_average_statistics([a, b])

        # np.mean of concatenated [a, b] arrays
        assert avg.mean == pytest.approx(np.mean([1.0, 3.0, 5.0, 7.0]))
        assert avg.median == pytest.approx(np.mean([2.0, 4.0, 6.0, 8.0]))
        assert avg.std == pytest.approx(np.mean([0.1, 0.2, 0.3, 0.4]))
        assert avg.min == pytest.approx(np.mean([0.0, 1.0, 4.0, 6.0]))
        assert avg.max == pytest.approx(np.mean([2.0, 5.0, 8.0, 9.0]))


class TestPadWithNan:
    def test_smaller_than_target_pads_with_nan(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = pad_with_nan(arr, (3, 3))
        assert out.shape == (3, 3)
        np.testing.assert_array_equal(out[:2, :2], arr)
        assert np.isnan(out[2, :]).all()
        assert np.isnan(out[:, 2]).all()

    def test_equal_shape_no_change(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = pad_with_nan(arr, arr.shape)
        np.testing.assert_array_equal(out, arr)

    def test_mixed_dimensions(self):
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = pad_with_nan(arr, (4, 3))
        assert out.shape == (4, 3)
        np.testing.assert_array_equal(out[:2], arr)
        assert np.isnan(out[2:]).all()


class TestComputeWithMethod:
    @pytest.fixture
    def arrays(self):
        return [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
        ]

    def test_mean(self, arrays):
        out = compute_with_method(arrays, np.mean)
        np.testing.assert_allclose(out, [[3.0, 4.0], [5.0, 6.0]])

    def test_median(self, arrays):
        out = compute_with_method(arrays, np.median)
        np.testing.assert_allclose(out, [[3.0, 4.0], [5.0, 6.0]])

    def test_nanmax(self, arrays):
        out = compute_with_method(arrays, np.nanmax)
        np.testing.assert_allclose(out, [[5.0, 6.0], [7.0, 8.0]])

    def test_nanmin(self, arrays):
        out = compute_with_method(arrays, np.nanmin)
        np.testing.assert_allclose(out, [[1.0, 2.0], [3.0, 4.0]])

    def test_unsupported_method_raises(self, arrays):
        with pytest.raises(ValueError):
            compute_with_method(arrays, np.sum)
