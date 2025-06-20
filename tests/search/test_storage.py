from copy import deepcopy
from datetime import datetime
import os
import h5py
import numpy as np
import pytest
from pulsar.search.storage import Storage, read_hdf5_metadata, count_files_with_extension
from pulsar.search.result import Result
from .fixtures import *

@pytest.fixture
def storage(tmpdir):
    output_path = str(tmpdir)
    return Storage(output_path)

class TestStorage:

    def test_missing_ids(self, storage, scenarios):
        test_cases = [
            {
                'title': 'Some scenarios are completed',
                'tod_ids': ['tod1', 'tod2', 'tod3'],
                'file_ids': [['tod1', 'tod2', 'tod3'], ['tod1']],
                'file_seeds': [scenarios[0].seed, scenarios[1].seed],
                'expected_missing_ids': ['tod2', 'tod3'],
                'expected_remaining_scenarios': scenarios[1:],
                'expected_completed_scenarios': scenarios[:1],
            },
            {
                'title': 'Missing files',
                'tod_ids': ['tod1', 'tod2', 'tod3'],
                'file_ids': [['tod1', 'tod2', 'tod3']],
                'file_seeds': [scenarios[0].seed],
                'expected_missing_ids': ['tod1', 'tod2', 'tod3'],
                'expected_remaining_scenarios': scenarios[1:],
                'expected_completed_scenarios': scenarios[:1],
            },
            {
                'title': 'All TOD IDs are missing',
                'tod_ids': ['tod1', 'tod2', 'tod3'],
                'file_ids': [[], []],
                'file_seeds': [scenarios[0].seed, scenarios[1].seed],
                'expected_missing_ids': ['tod1', 'tod2', 'tod3'],
                'expected_remaining_scenarios': scenarios,
                'expected_completed_scenarios': [],
            },
            {
                'title': 'Some TOD IDs are missing from all files',
                'tod_ids': ['tod1', 'tod2', 'tod3'],
                'file_ids': [['tod1'], ['tod1']],
                'file_seeds': [scenarios[0].seed, scenarios[1].seed],
                'expected_missing_ids': ['tod2', 'tod3'],
                'expected_remaining_scenarios': scenarios,
                'expected_completed_scenarios': [],
            },
            {
                'title': 'Some TOD IDs are missing from different files',
                'tod_ids': ['tod1', 'tod2', 'tod3'],
                'file_ids': [['tod1', 'tod2'], ['tod2', 'tod3']],
                'file_seeds': [scenarios[0].seed, scenarios[1].seed],
                'expect_error': True,
            },
            {
                'title': 'Some scenarios are completed and some are missing',
                'tod_ids': ['tod1', 'tod2', 'tod3'],
                'file_ids': [['tod1', 'tod2'], []],
                'file_seeds': [scenarios[0].seed, 123456789],
                'expect_error': True,
            },
            {
                'title': 'No TOD IDs are missing',
                'tod_ids': ['tod1', 'tod2', 'tod3'],
                'file_ids': [['tod1', 'tod2', 'tod3'], ['tod1', 'tod2', 'tod3']],
                'file_seeds': [scenarios[0].seed, scenarios[1].seed],
                'expected_missing_ids': [],
                'expected_remaining_scenarios': [],
                'expected_completed_scenarios': scenarios,
            },
            {
                'title': 'Non-unique scenario seeds in files',
                'tod_ids': ['tod1', 'tod2', 'tod3'],
                'file_ids': [[], []],
                'file_seeds': [scenarios[0].seed, scenarios[0].seed],
                'expect_error': True,
            },
            {
                'title': 'Existing scenario seeds in files',
                'tod_ids': ['tod1', 'tod2', 'tod3'],
                'file_ids': [[], [], ['tod1', 'tod2', 'tod3']],
                'file_seeds': [scenarios[0].seed, scenarios[1].seed, 123456789],
                'expected_missing_ids': ['tod1', 'tod2', 'tod3'],
                'expected_remaining_scenarios': scenarios,
                'expected_completed_scenarios': [],
            }
        ]

        for test_case in test_cases:
            filenames = []
            for i, file_ids in enumerate(test_case['file_ids']):
                file_path = os.path.join(storage.output_path, f'file{i}.hdf5')
                filenames.append(file_path)
                with h5py.File(file_path, 'w') as hdf:
                    if ('file_seeds' in test_case and i < len(test_case['file_seeds'])
                        and test_case['file_seeds'][i] is not None):
                        hdf.attrs['scenario_seed'] = test_case['file_seeds'][i]
                    for id in file_ids:
                        hdf.create_group(id)

            if 'expect_error' in test_case:
                with pytest.raises(ValueError):
                    storage.missing_ids(test_case['tod_ids'], scenarios)
                continue

            remaining, missing_ids, completed = storage.missing_ids(test_case['tod_ids'], scenarios)

            assert missing_ids == test_case['expected_missing_ids'], test_case['title']
            assert remaining == test_case['expected_remaining_scenarios'], test_case['title']
            assert completed == test_case['expected_completed_scenarios'], test_case['title']

            for file_path in filenames:
                os.remove(file_path)

    def test_update(self, storage, scenario):
        dt = datetime.strptime('2021-01-01T00:00:00', '%Y-%m-%dT%H:%M:%S')

        results = [
            Result(
                datetime=dt,
                tod_id='tod1',
                scenario=scenario,
                rhs=[np.array([1, 2, 3])],
                div=[np.array([4, 5, 6])]
            ),
            Result(
                datetime=dt,
                tod_id='tod2',
                scenario=scenario,
                rhs=[np.array([7, 8, 9])],
                div=[np.array([10, 11, 12])],
            ),
        ]

        # test file with existing tod id
        file_path = os.path.join(storage.output_path, results[1].filename)
        with h5py.File(file_path, 'w') as hdf:
            group = hdf.create_group('tod2')
            group.create_dataset('rhs', data=np.array([[7, 8, 9]]))
            group.create_dataset('div', data=np.array([[10, 11, 12]]))

        storage.update(results)

        file_path = os.path.join(storage.output_path, results[0].filename)
        with h5py.File(file_path, 'r') as hdf:
            assert hdf.attrs['id'] == 2
            assert hdf.attrs['date'] == '2021-01-01'
            assert hdf.attrs['title'] == 'Title'
            assert hdf.attrs['target']
            assert hdf.attrs['scenario_seed'] == 1
            assert hdf.attrs['operations']
            assert hdf.attrs['search_profiles']
            assert hdf.attrs['filter']
            assert hdf.attrs['noise']
            assert hdf.attrs['polarization_components'] == 3

            assert 'tod1' in hdf
            assert 'tod2' in hdf

            tod1_group = hdf['tod1']
            assert np.array_equal(tod1_group['rhs'][:], np.array([[1, 2, 3]]))
            assert np.array_equal(tod1_group['div'][:], np.array([[4, 5, 6]]))

            tod2_group = hdf['tod2']
            assert np.array_equal(tod2_group['rhs'][:], np.array([[7, 8, 9]]))
            assert np.array_equal(tod2_group['div'][:], np.array([[10, 11, 12]]))

        file_path = os.path.join(storage.output_path, results[1].filename)
        with h5py.File(file_path, 'r') as hdf:
            assert hdf.attrs['id'] == 2
            assert hdf.attrs['date'] == '2021-01-01'
            assert hdf.attrs['title'] == 'Title'
            assert hdf.attrs['target']
            assert hdf.attrs['scenario_seed'] == 1
            assert hdf.attrs['operations']
            assert hdf.attrs['search_profiles']
            assert hdf.attrs['filter']
            assert hdf.attrs['noise']
            assert hdf.attrs['polarization_components'] == 3

            assert 'tod1' in hdf
            assert 'tod2' in hdf

            tod1_group = hdf['tod1']
            assert np.array_equal(tod1_group['rhs'][:], np.array([[1, 2, 3]]))
            assert np.array_equal(tod1_group['div'][:], np.array([[4, 5, 6]]))

            tod2_group = hdf['tod2']
            assert np.array_equal(tod2_group['rhs'][:], np.array([[7, 8, 9]]))
            assert np.array_equal(tod2_group['div'][:], np.array([[10, 11, 12]]))

        scenario3 = deepcopy(scenario)
        scenario3.seed = 3
        scenario3.filter = None
        scenario3.metadata = {'tuple': (1, 2, 3)}

        results = [
            Result(
                datetime=dt,
                tod_id='tod3',
                scenario=scenario3,
                rhs=[np.zeros((3,))],
                div=[np.zeros((3,3))],
            )
        ]

        storage.update(results)

        file_path = os.path.join(storage.output_path, results[0].filename)
        with h5py.File(file_path, 'r') as hdf:
            assert hdf.attrs['filter'] == 'None'

    def test_read(self, storage):
        file_path = os.path.join(storage.output_path, 'file1.hdf5')
        with h5py.File(file_path, 'w') as hdf:
            hdf.attrs['key1'] = 'value1'
            hdf.attrs['key2'] = np.array([1, 2, 3])
            hdf.create_group('tod1')
            tod1_group = hdf['tod1']
            tod1_group.create_dataset('rhs', data=np.array([4, 5, 6]))
            tod1_group.create_dataset('div', data=np.array([[7, 8, 9], [7, 8, 9], [7, 8, 9]]))

        data, metadata, references = storage.read(file_path)

        assert np.array_equal(data['tod_id'], np.array(['tod1']))
        assert np.array_equal(data['rhs'], np.array([[4, 5, 6]]))
        assert np.array_equal(data['div'], np.array([[[7, 8, 9], [7, 8, 9], [7, 8, 9]]]))

        assert metadata == {'key1': 'value1', 'key2': (1, 2, 3)}

        assert references.to_dict(orient='list') == {
            'tod_id': ['tod1'],
            'rhs_shape': [(3,)],
            'div_shape': [(3, 3)],
        }

    def test_metadata_from_directory(self, storage):
        directory = storage.output_path

        # Create some dummy HDF5 files
        file1 = os.path.join(directory, 'file1.hdf5')
        with h5py.File(file1, 'w') as hdf:
            hdf.attrs['id'] = 1
            hdf.attrs['title'] = 'Title 1'
            hdf.attrs['target'] = 'Target 1'
            hdf.attrs['scenario_seed'] = 1
            hdf.create_group('tod1')

        file2 = os.path.join(directory, 'file2.hdf5')
        with h5py.File(file2, 'w') as hdf:
            hdf.attrs['id'] = 2
            hdf.attrs['title'] = 'Title 2'
            hdf.attrs['target'] = 'Target 2'
            hdf.attrs['scenario_seed'] = 2
            hdf.create_group('tod2')
            hdf.create_group('tod3')

        df = storage.metadata_from_directory(directory)

        assert df.to_dict(orient='list') == {
            'file': [file1, file2],
            'date': ['', ''],
            'id': [1, 2],
            'title': ['Title 1', 'Title 2'],
            'target': ['Target 1', 'Target 2'],
            'scenario_seed': [1, 2],
            'search_profiles': ['', ''],
            'operations': ['', ''],
            'polarization_components': ['', ''],
            'comments': ['', ''],
            'tods': [1, 2],
        }

    def test_read_hdf5_metadata(self):
        attrs = {
            'key1': 'value1',
            'key2': np.array([1, 2, 3]),
            'key3': 'None'
        }

        metadata = read_hdf5_metadata(attrs)

        assert metadata == {'key1': 'value1', 'key2': (1, 2, 3), 'key3': None}

def test_count_files_with_extension(storage):
    directory = storage.output_path

    # Create some dummy files
    file1 = os.path.join(directory, 'file1.txt')
    with open(file1, 'w') as f:
        f.write('')

    file2 = os.path.join(directory, 'file2.txt')
    with open(file2, 'w') as f:
        f.write('')

    file3 = os.path.join(directory, 'file3.txt')
    with open(file3, 'w') as f:
        f.write('')

    count = count_files_with_extension(directory, '.txt')

    assert count == 3

    # invalid directory
    with pytest.raises(ValueError):
        count_files_with_extension('invalid_directory', '.txt')