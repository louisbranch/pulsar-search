import sys
import pytest
from unittest.mock import MagicMock, patch

# Mock the dependencies
mocks = {
    'enact': MagicMock(),
    'enact.actdata': MagicMock(),
    'enact.actscan': MagicMock(),
    'enact.cuts': MagicMock(),
    'enact.filedb': MagicMock(),
    'enact.nmat_measure': MagicMock(),
    'enlib': MagicMock(),
    'enlib.config': MagicMock(),
    'enlib.dataset': MagicMock(),
    'enlib.errors': MagicMock(),
    'enlib.fft': MagicMock(),
    'enlib.gapfill': MagicMock(),
    'enlib.sampcut': MagicMock(),
    'enlib.pmat': MagicMock(),
    'enlib.utils': MagicMock(),
    'mpi4py': MagicMock(),
    'mpi4py.MPI': MagicMock(),
}

@pytest.fixture(scope='session', autouse=True)
def apply_mocks():
    """Apply all the necessary mocks before any test is run."""
    from .test_mocks import create_mock_instrument
    from pulsar.config import config
    config.instrument = create_mock_instrument()
    with patch.dict('sys.modules', mocks):
        yield

def pytest_configure(config):
    """Apply all the necessary mocks before any test modules are imported."""
    patcher = patch.dict(sys.modules, mocks)
    patcher.start()
    config.add_cleanup(patcher.stop)