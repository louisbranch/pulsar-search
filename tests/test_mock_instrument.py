import numpy as np
import pytest
from numpy.random import default_rng

from pulsar.mock_instrument.instrument import Config, MockInstrument
from pulsar.mock_instrument.instrument import TOD as MockTOD


@pytest.fixture
def small_instrument():
    return MockInstrument(
        max_tods=4,
        dets_per_tod=2,
        samples_per_tod=64,
        sampling_rate=10.0,
    )


class TestMockInstrument:
    def test_tods_honors_limit(self, small_instrument):
        out = list(small_instrument.tods('unused', limit=2))
        assert len(out) == 2

    def test_tods_filters_by_ids(self, small_instrument):
        out = list(small_instrument.tods('unused', ids=['1', '3']))
        assert [t.id for t in out] == ['1', '3']

    def test_tods_honors_dets(self, small_instrument):
        out = list(small_instrument.tods('unused', limit=1, dets=[0, 1, 2]))
        assert len(out) == 1
        assert out[0].num_detectors == 3

    def test_tod_ids_returns_max_tods(self, small_instrument):
        ids = small_instrument.tod_ids('unused')
        assert ids == ['0', '1', '2', '3']

    def test_pointing_model_returns_none(self, small_instrument):
        assert small_instrument.pointing_model() is None


class TestMockTOD:
    def test_data_lazily_generates(self):
        tod = MockTOD(
            id='x',
            num_detectors=2,
            num_samples=32,
            sampling_rate=10.0,
            rng=default_rng(seed=0),
        )
        assert tod._data is None
        data = tod.data
        assert data.shape == (2, 32)
        assert tod.data is data

    def test_data_setter_overwrites(self):
        tod = MockTOD(
            id='x',
            num_detectors=2,
            num_samples=4,
            sampling_rate=10.0,
        )
        new = np.zeros((2, 4))
        tod.data = new
        np.testing.assert_array_equal(tod.data, new)

    def test_generate_signal_within_plausible_bounds(self):
        cfg = Config(
            one_over_f_noise=10.0,
            gaussian_noise_std=1.0,
            amplitude_range=(0.5, 1.5),
            trend_amplitude=10.0,
            trend_frequency=0.01,
        )
        tod = MockTOD(
            id='x',
            num_detectors=2,
            num_samples=128,
            sampling_rate=10.0,
            rng=default_rng(seed=42),
            config=cfg,
        )
        data = tod.data
        assert data.shape == (2, 128)
        assert np.isfinite(data).all()
        assert np.all(np.abs(data.mean(axis=1)) < 50.0)
