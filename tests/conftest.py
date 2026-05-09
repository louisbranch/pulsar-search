import sys
import pytest
from unittest.mock import patch

from ._act_stubs import build_mocks

# Build once at conftest import so the same module objects are shared by
# every patch.dict invocation. Tests can patch attributes on these stubs
# (e.g. patch.object(filedb, 'data', {...})) and observe state across both
# pytest_configure and the session fixture.
mocks = build_mocks()


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
