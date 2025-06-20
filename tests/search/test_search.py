import pytest
from unittest.mock import Mock
from pulsar import ConstantProfile, Target
from pulsar.search import Search, Scenario
from .fixtures import *

@pytest.fixture
def scenario(target):
    return Scenario(
        title='test_scenario',
        target=target,
        search_profiles=[ConstantProfile()],
    )

@pytest.fixture
def search(tmpdir, scenario):
    return Search(
        scenarios=[scenario],
        tod_path='tests/test_data/tod.txt',
        output_path=tmpdir,
    )

class TestSearch:
    def test_init(self, scenario):
        mock_extension = Mock()

        search = Search(
            scenarios=[scenario],
            tod_path='tests/test_data/tod.txt',
            output_path='tests/test_data',
            extensions=[mock_extension]
        )
        assert isinstance(search, Search)

        assert search.instrument is not None
        assert search.pointing_model is not None
        assert search._extension_manager.extensions == [mock_extension]

    def test_run(self, search):
        # Sequential search
        search.parallel = False
        search.run()

        # Parallel search
        from mpi4py import MPI
        MPI.COMM_WORLD.recv.return_value = None # STOP worker signal
        search.parallel = True
        search.run()