import logging
import sys
from unittest.mock import patch, Mock
from .test_mocks import create_mock_instrument

import pulsar

def instrument():
    return pulsar.config.config.instrument

def log():
    return pulsar.log.log.logger

def test_config_init():
    pulsar.init(create_mock_instrument(), log_level=logging.INFO)
    assert isinstance(instrument(), Mock)
    assert log().level == logging.INFO

def test_config_init_with_instrument():
    mock = create_mock_instrument()
    pulsar.init(mock)
    assert instrument() == mock
    assert log().level == logging.WARNING

def test_config_init_with_log_level():
    pulsar.init(log_level=logging.INFO)
    assert instrument() is not None
    assert log().level == logging.INFO

def test_config_init_with_instrument_and_log_level():
    mock = create_mock_instrument()
    pulsar.init(mock, log_level=logging.DEBUG)
    assert instrument() == mock
    assert log().level == logging.DEBUG

@patch('pulsar.log.warning')
def test_config_init_warning(mock_warning):
    # Ensure the submodule is not already imported
    if 'pulsar.act' in sys.modules:
        del sys.modules['pulsar.act']
    if 'pulsar' in sys.modules:
        del sys.modules['pulsar']
    
    # Patch sys.modules to simulate ImportError
    with patch.dict('sys.modules', {'pulsar.act': None}):
        import importlib
        import pulsar
        importlib.reload(pulsar)
        pulsar.init()
        
        # Check the behavior of pulsar.ACT
        assert instrument() is None

        # Verify that the warning was logged
        mock_warning.assert_called_once_with("No instrument provided. Most functions will not work.")

def test_init():
    instrument = create_mock_instrument()
    pulsar.init(instrument, log_level=logging.INFO)
    assert pulsar.config.config.instrument == instrument
    assert log().level == logging.INFO


def test_instrument_accessor_returns_active_instrument():
    """`pulsar.instrument()` is the public accessor for the current instrument
    on the global config — verify it tracks `pulsar.init()`."""
    mock = create_mock_instrument()
    pulsar.init(mock)
    assert pulsar.instrument() is mock