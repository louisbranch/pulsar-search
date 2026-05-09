import os
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest

from pulsar import Target
from pulsar.targeting.targeting import (
    Targeting,
    append_to_hdf5_file,
    read_hdf5_file,
    target_hash,
)
from ..test_mocks import create_mock_tod


@pytest.fixture
def target_a():
    return Target(name='A', ra=0.5, dec=0.3, radius=0.1, T=1.0)


@pytest.fixture
def target_b():
    return Target(name='B', ra=1.0, dec=0.5, radius=0.2, T=1.0)


class TestTargetHash:
    def test_stable_for_identical_attributes(self, target_a):
        other = Target(name='different-name', ra=0.5, dec=0.3, radius=0.1, T=2.0)
        assert target_hash(target_a) == target_hash(other)

    def test_differs_for_different_position(self, target_a, target_b):
        assert target_hash(target_a) != target_hash(target_b)


class TestReadHdf5File:
    def test_returns_all_missing_when_file_does_not_exist(self, tmpdir):
        path = os.path.join(str(tmpdir), 'missing.hdf5')
        result = read_hdf5_file(path, ['t1', 't2'], ['target1', 'target2'])
        assert result == {'t1': ['target1', 'target2'], 't2': ['target1', 'target2']}

    def test_returns_all_missing_when_groups_absent(self, tmpdir):
        path = os.path.join(str(tmpdir), 'empty.hdf5')
        with h5py.File(path, 'w') as f:
            f.attrs['placeholder'] = 1
        result = read_hdf5_file(path, ['t1'], ['target1'])
        assert result == {'t1': ['target1']}

    def test_returns_only_missing_targets_per_tod(self, tmpdir):
        path = os.path.join(str(tmpdir), 'partial.hdf5')
        with h5py.File(path, 'w') as f:
            f.create_group('/targets')
            tods_group = f.create_group('/tods')
            t1 = tods_group.create_group('t1')
            t1.create_group('target1')  # only target1 is recorded for t1
            # t2 is entirely absent
        result = read_hdf5_file(path, ['t1', 't2'], ['target1', 'target2'])
        assert result == {
            't1': ['target2'],
            't2': ['target1', 'target2'],
        }


class TestAppendToHdf5File:
    def test_creates_groups_on_first_call(self, tmpdir, target_a):
        path = os.path.join(str(tmpdir), 'append.hdf5')
        ranges = [np.array([[0, 10], [20, 30]])]

        append_to_hdf5_file(path, 'tod_x', [target_a], ranges)

        with h5py.File(path, 'r') as f:
            assert '/targets' in f
            assert '/tods/tod_x' in f
            tid = target_hash(target_a)
            assert tid in f['/targets']
            assert f['/targets'][tid].attrs['name'] == 'A'
            assert tid in f['/tods/tod_x']
            np.testing.assert_array_equal(
                f['/tods/tod_x'][tid]['samples_range'][:], ranges[0]
            )

    def test_second_call_reuses_existing_groups(self, tmpdir, target_a, target_b):
        path = os.path.join(str(tmpdir), 'append.hdf5')
        append_to_hdf5_file(path, 'tod_x', [target_a], [np.array([[0, 1]])])
        # Different tod, different target — should not error or duplicate.
        append_to_hdf5_file(path, 'tod_y', [target_b], [np.array([[2, 3]])])

        with h5py.File(path, 'r') as f:
            assert target_hash(target_a) in f['/targets']
            assert target_hash(target_b) in f['/targets']
            assert 'tod_x' in f['/tods']
            assert 'tod_y' in f['/tods']


class TestTargeting:
    def test_post_init_falls_back_to_config(self, target_a, tmpdir):
        # `targeting` captured a reference to the config object at import time,
        # so check against that reference rather than `pulsar.config.config`,
        # which gets re-bound by `pulsar.init` in other tests.
        from pulsar.targeting import targeting as targeting_mod
        t = Targeting(targets=[target_a], tod_path='unused', output_path=str(tmpdir))
        assert t.instrument is targeting_mod.config.instrument
        # pointing_model is also pulled from config (whatever the mock returns).
        assert t.pointing_model is not None

    def test_search_targets_returns_tod_id_and_ranges(self, target_a, tmpdir):
        tod = create_mock_tod(id='tod_x')
        samples = MagicMock(name='Samples')
        samples.ranges = [(0, 10), (20, 30)]
        tod.locate_source.side_effect = lambda ra, dec, R: (None, samples)

        t = Targeting(targets=[target_a], tod_path='unused', output_path=str(tmpdir))
        tod_id, results = t._search_targets(tod, [target_a])

        assert tod_id == 'tod_x'
        assert results == [[(0, 10), (20, 30)]]
        tod.calibrate.assert_called_once()
