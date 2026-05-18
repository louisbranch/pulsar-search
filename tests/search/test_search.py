import ast
import inspect
import logging
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pulsar import ConstantProfile
from pulsar.search import Search, Scenario
from pulsar.search import search as search_module
from .fixtures import *


def test_mpi4py_import_is_deferred():
    """Regression: `pulsar.search.search` must not pull in `mpi4py` at
    module-load time. The import is deferred to `_load_mpi()` so the module
    (and `Search` sequential mode) can be used without the `[parallel]` extra.
    """
    tree = ast.parse(inspect.getsource(search_module))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith('mpi4py'), (
                f"mpi4py must not be imported at module scope; found "
                f"`from {node.module} import ...` at line {node.lineno}"
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith('mpi4py'), (
                    f"mpi4py must not be imported at module scope; found "
                    f"`import {alias.name}` at line {node.lineno}"
                )


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
        mock_extension = MagicMock()

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

    def test_post_init_creates_missing_output_path(self, tmpdir, scenario):
        new_dir = os.path.join(str(tmpdir), 'subdir', 'results')
        assert not os.path.exists(new_dir)
        Search(
            scenarios=[scenario],
            tod_path='unused',
            output_path=new_dir,
        )
        assert os.path.isdir(new_dir)

    def test_setup_search_uses_instrument_tod_ids_when_none(self, tmpdir, scenario):
        s = Search(
            scenarios=[scenario],
            tod_path='unused',
            output_path=str(tmpdir),
            tod_ids=None,
        )
        s.instrument = MagicMock()
        s.instrument.tod_ids.return_value = ['a', 'b']
        ids, scenarios, storage = s._setup_search()
        s.instrument.tod_ids.assert_called_once_with('unused')
        assert ids == ['a', 'b']

    def test_setup_search_uses_explicit_tod_ids(self, tmpdir, scenario):
        s = Search(
            scenarios=[scenario],
            tod_path='unused',
            output_path=str(tmpdir),
            tod_ids=['x', 'y'],
        )
        s.instrument = MagicMock()
        ids, scenarios, storage = s._setup_search()
        s.instrument.tod_ids.assert_not_called()
        assert ids == ['x', 'y']

    def test_setup_search_honors_tod_limit(self, tmpdir, scenario):
        s = Search(
            scenarios=[scenario],
            tod_path='unused',
            output_path=str(tmpdir),
            tod_ids=['a', 'b', 'c', 'd'],
            tod_limit=2,
        )
        ids, scenarios, storage = s._setup_search()
        assert ids == ['a', 'b']

    def test_setup_search_save_false_returns_null_storage(self, tmpdir, scenario):
        from pulsar.search.storage import Storage
        s = Search(
            scenarios=[scenario],
            tod_path='unused',
            output_path=str(tmpdir),
            tod_ids=['a', 'b'],
            save=False,
        )
        ids, scenarios_out, storage = s._setup_search()
        assert ids == ['a', 'b']
        assert scenarios_out == [scenario]
        assert isinstance(storage, Storage)
        assert storage.output_path == '/dev/null'

    def test_setup_search_returns_only_missing_ids_for_resume(self, tmpdir, scenario):
        # Pre-seed the output dir with a partial result file so the resume path
        # in _setup_search filters out already-completed TOD ids.
        import h5py
        result_path = os.path.join(str(tmpdir), f'_{scenario.seed}.hdf5')
        with h5py.File(result_path, 'w') as hdf:
            hdf.attrs['scenario_seed'] = scenario.seed
            hdf.create_group('tod1')  # tod1 is done; tod2 and tod3 remain.

        s = Search(
            scenarios=[scenario],
            tod_path='unused',
            output_path=str(tmpdir),
            tod_ids=['tod1', 'tod2', 'tod3'],
        )
        ids, scenarios_out, storage = s._setup_search()
        assert ids == ['tod2', 'tod3']
        assert scenarios_out == [scenario]

    def test_setup_search_skips_fully_completed_scenarios(self, tmpdir, scenario):
        import h5py
        # All three TOD ids are present -> the scenario is complete and
        # _setup_search should report nothing to do.
        result_path = os.path.join(str(tmpdir), f'_{scenario.seed}.hdf5')
        with h5py.File(result_path, 'w') as hdf:
            hdf.attrs['scenario_seed'] = scenario.seed
            for tid in ('tod1', 'tod2', 'tod3'):
                hdf.create_group(tid)

        s = Search(
            scenarios=[scenario],
            tod_path='unused',
            output_path=str(tmpdir),
            tod_ids=['tod1', 'tod2', 'tod3'],
        )
        ids, scenarios_out, storage = s._setup_search()
        assert ids == []
        assert scenarios_out == []

    def test_run_sequential_exits_early_when_no_scenarios(self, tmpdir, scenario, caplog):
        s = Search(
            scenarios=[scenario],
            tod_path='unused',
            output_path=str(tmpdir),
            tod_ids=['a'],
        )
        # Pretend the storage already covers the work
        with patch.object(Search, '_setup_search', return_value=([], [], MagicMock())):
            with caplog.at_level(logging.INFO, logger='pulsar.log.log'):
                s.run_sequential()
        assert any('No scenarios to process' in r.message for r in caplog.records)

    def test_search_scenarios_continues_after_exception(self, tmpdir, scenario, target):
        s = Search(
            scenarios=[scenario],
            tod_path='unused',
            output_path=str(tmpdir),
        )
        with patch.object(Search, '_run_search', side_effect=RuntimeError('boom')):
            results = s._search_scenarios(MagicMock(id='tod1'), [scenario])
        assert results == []  # error logged, no result appended

    def test_search_scenarios_deepcopies_when_multiple_scenarios(self, tmpdir, target):
        scenarios = [
            Scenario(title='A', target=target, search_profiles=[ConstantProfile()]),
            Scenario(title='B', target=target, search_profiles=[ConstantProfile()]),
        ]
        s = Search(
            scenarios=scenarios,
            tod_path='unused',
            output_path=str(tmpdir),
        )

        original = MagicMock(id='tod1')
        seen_ids = []
        def fake_run(tod, scenario):
            seen_ids.append(id(tod))
            return np.zeros((1, 3)), np.zeros((1, 3, 3))

        with patch.object(Search, '_run_search', side_effect=fake_run):
            results = s._search_scenarios(original, scenarios)

        assert len(results) == 2
        # Each scenario received a different tod object (deep copies) and neither was the original.
        assert id(original) not in seen_ids
        assert seen_ids[0] != seen_ids[1]