import logging
from unittest.mock import patch

import pytest

from pulsar.log.log import *

def test_log_init():
    with patch('logging.Logger.hasHandlers') as mock_has_handlers:
        mock_has_handlers.return_value = False
        log = Log(logging.INFO)
        assert log.logger.level == logging.INFO

def test_log_print():
    log = Log(logging.INFO)
    with patch('logging.Logger.log') as mock_log:
        log.print(logging.ERROR, "Error message")
        mock_log.assert_called_once_with(logging.ERROR, "Error message")

def test_log_utc_time():
    log = Log(logging.INFO)
    utc_time = log._utc_time()
    assert isinstance(utc_time, tuple)
    assert len(utc_time) == 9

def test_log_error():
    with patch('logging.Logger.log') as mock_log:
        error("Error message")
        mock_log.assert_called_once_with(logging.ERROR, "Error message")

def test_log_critical():
    with patch('logging.Logger.log') as mock_log:
        with pytest.raises(SystemExit):
            critical("Critical message")
        # Assertions run after SystemExit is raised and caught.
        mock_log.assert_called_once_with(logging.CRITICAL, "Critical message")

def test_log_info():
    with patch('logging.Logger.log') as mock_log:
        info("Info message")
        mock_log.assert_called_once_with(logging.INFO, "Info message")

def test_log_debug():
    with patch('logging.Logger.log') as mock_log:
        debug("Debug message")
        mock_log.assert_called_once_with(logging.DEBUG, "Debug message")

def test_log_warning():
    with patch('logging.Logger.log') as mock_log:
        warning("Warning message")
        mock_log.assert_called_once_with(logging.WARNING, "Warning message")