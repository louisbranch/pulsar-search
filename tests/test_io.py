import pulsar
from pulsar import io as pulsar_io

from .test_mocks import create_mock_instrument


def _swap_instrument():
    """Replace the instrument that `pulsar.io` resolves at call time.

    `pulsar.io` captured a reference to the original `config` object at import,
    so mutating `pulsar.config.config.instrument` (which `pulsar.init` re-binds
    in other tests) would not affect calls through `pulsar_io`. Always reach
    through the io module's own reference.
    """
    instrument = create_mock_instrument()
    pulsar_io.config.instrument = instrument
    return instrument


def test_tods():
    result = pulsar.tods("/path/to/tods")
    assert len(list(result)) == 5


def test_tod_ids():
    result = pulsar.tod_ids("/path/to/tods")
    assert result == ["mock_tod_1", "mock_tod_2", "mock_tod_3"]


def test_tods_forwards_filters_to_instrument():
    instrument = _swap_instrument()
    list(pulsar_io.tods("/path/to/tods", limit=3, dets=[1, 2], ids=["mock_tod_1"]))
    instrument.tods.assert_called_with("/path/to/tods", limit=3, dets=[1, 2], ids=["mock_tod_1"])


def test_tod_ids_forwards_to_instrument():
    instrument = _swap_instrument()
    pulsar_io.tod_ids("/path/to/tods")
    instrument.tod_ids.assert_called_with("/path/to/tods")
