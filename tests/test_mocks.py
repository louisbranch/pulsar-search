import numpy as np
from typing import List
from unittest.mock import Mock
from pulsar.pointing_models import PointingModel
from pulsar.tod import TOD
from pulsar.instrument import Instrument

# Mock TOD
def create_mock_tod(id='mock_tod_1') -> Mock:
    mock_tod = Mock(spec=TOD, name='MockTOD')
    mock_tod.id = id
    mock_tod.data = np.random.randn(10, 100)
    mock_tod.num_detectors = 10
    mock_tod.num_samples = 100
    mock_tod.sampling_rate = 400
    mock_tod.calibrate.side_effect = lambda: None
    mock_tod.calibrated = True
    mock_tod.downsample.side_effect = lambda factor: None
    mock_tod.remove_source.side_effect = lambda ra, dec, R: None
    mock_tod.locate_source.side_effect = lambda ra, dec, R: (np.random.randn(100), np.random.choice([True, False], 100))
    mock_tod.multiplicative_gaussian_noise.side_effect = lambda sigma: None
    mock_tod.fill_gaps.side_effect = lambda data=None: None
    return mock_tod

# Mock PointingModel
def create_mock_pointing_model(ncomp=3) -> Mock:
    mock_pointing_model = Mock(spec=PointingModel, name='MockPointingModel')
    mock_pointing_model.forward.side_effect = lambda data, amps, pmul=1: None
    mock_pointing_model.backward.side_effect = lambda data, pmul=1: np.random.randn(1, ncomp, ncomp)
    return mock_pointing_model

# Mock Instrument
def create_mock_instrument() -> Mock:
    def mock_tod_generator(path, limit=0, dets=None, ids=None):
        for i in range(5):
            yield create_mock_tod(id=f"mock_tod_{i+1}")

    def mock_pointing_model(*args, **kwargs):
        return lambda tods, sources, ncomp=3: create_mock_pointing_model(ncomp)

    mock_instrument = Mock(spec=Instrument, name='MockInstrument')
    mock_instrument.tods.side_effect = mock_tod_generator
    mock_instrument.tod_ids.side_effect = lambda path: ["mock_tod_1", "mock_tod_2", "mock_tod_3"]
    mock_instrument.pointing_model.side_effect = mock_pointing_model
    return mock_instrument