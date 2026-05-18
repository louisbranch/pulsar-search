import os
from typing import List, Tuple

import h5py
import json
import numpy as np
import pandas as pd

from .result import Result
from .scenario import Scenario
from .. import log

class Storage:
    """
    Storage is used to save them to disk in an HDF5 file.
    It can also read the result from an existing HDF5 file, including the metadata and references.
    """
    output_path: str
    extension: str = '.hdf5'

    def __init__(self, output_path: str = '.'):
        self.output_path = output_path

    def missing_ids(self, tod_ids: List[str], scenarios: List[Scenario]) -> Tuple[List[Scenario], List[str], List[Scenario]]:
        """
        Check if the search results for the given TOD IDs and scenarios already exist in the output directory.
        Return the scenarios with missing TOD IDs, the missing TOD IDs for those scenarios,
        and the scenarios where all TOD IDs are present.

        Parameters:
        - tod_ids (List[str]): A list of TOD IDs to check.
        - scenarios (List[Scenario]): A list of scenarios to check.

        Returns:
        - Tuple[List[Scenario], List[str], List[Scenario]]: A tuple containing:
            - List[Scenario]: A list of scenarios that have missing TOD IDs.
            - List[str]: A list of TOD IDs that are missing from the search results for those scenarios.
            - List[Scenario]: A list of scenarios that have all TOD IDs present.
        """

        scenarios_seeds = {scenario.seed: scenario for scenario in scenarios}
        seeds_found = set()
        missing_ids = set()
        incomplete_seeds = set()

        # Iterate over all files in the directory
        for filename in os.listdir(self.output_path):
            if not filename.endswith(self.extension):
                continue

            file_path = os.path.join(self.output_path, filename)
            with h5py.File(file_path, 'r') as hdf:
                # Check if the scenario_seed matches one of the given scenario_seeds
                seed = int(hdf.attrs['scenario_seed'])
                if seed in seeds_found:
                    raise ValueError(f'Scenario seed {seed} is not unique.')
                if seed not in scenarios_seeds:
                    continue

                seeds_found.add(seed)
                ids = set(hdf.keys())

                # Check if all TOD IDs are present in the file
                file_missing_ids = set(tod_ids) - ids
                if file_missing_ids == set():
                    continue

                if missing_ids == set():
                    missing_ids = file_missing_ids
                elif file_missing_ids != missing_ids:
                    raise ValueError(f'The missing TOD IDs are not consistent across files.')
                
                incomplete_seeds.add(seed)

        remaining_scenarios = []
        completed_scenarios = []

        for scenario in scenarios:
            if scenario.seed not in seeds_found:
                if missing_ids == set():
                    remaining_scenarios.append(scenario)
                    continue
                else:
                    log.debug(f'Scenario {scenario.seed} is missing TOD IDs: {missing_ids}')
                    raise ValueError(f'Starting a new scenario while some TOD IDs are missing in other scenarios.')

            if scenario.seed in incomplete_seeds:
                remaining_scenarios.append(scenario)
            else:
                completed_scenarios.append(scenario)
            
        if len(remaining_scenarios) == 0:
            remaining_ids = []
        else:
            remaining_ids = sorted(list(missing_ids)) if missing_ids != set() else tod_ids

        return remaining_scenarios, remaining_ids, completed_scenarios

    def _serialize_metadata(self, metadata: dict) -> dict:
        """
        Serialize the metadata dictionary to compatible formats for storage in an HDF5 file.
        """
        serialized_metadata = {}
        for key, value in metadata.items():
            if value is None:
                value = 'None'
            elif isinstance(value, tuple):
                value = np.array(value)
            elif isinstance(value, dict) or isinstance(value, list):
                value = json.dumps(value)
            serialized_metadata[key] = value
            log.debug(f'Serialized metadata: {key} = {value}')
        return serialized_metadata

    def update(self, results: List[Result]):
        """
        Append the search results to the HDF5 file in the output directory.
        If no file exists, it will be created and the metadata will be added.
        """
        for i, result in enumerate(results):

            metadata = result.metadata
            filename = result.filename
            filepath = os.path.join(self.output_path, filename)

            with h5py.File(filepath, 'a') as hdf_file:
                # Add metadata if not already present
                if 'id' not in hdf_file.attrs:
                    id = count_files_with_extension(self.output_path) + 1
                    log.debug(f'Creating search results to {filepath} with ID {id}')

                    hdf_file.attrs['id'] = id
                    for key, value in self._serialize_metadata(metadata).items():
                        hdf_file.attrs[key] = value
                else:
                    log.debug(f'Appending search results to {filepath}')

                if result.tod_id in hdf_file:
                    log.warning(f'TOD ID {result.tod_id} already exists in {filename}. Skipping...')
                    continue

                # Create a group for the current TOD if it doesn't exist
                tod_group = hdf_file.require_group(result.tod_id)

                # Combine all rhs and div matrices into separate arrays
                rhs = np.stack(result.rhs, axis=0)
                div = np.stack(result.div, axis=0)

                # Create datasets for combined rhs and div
                tod_group.create_dataset('rhs', data=rhs)
                tod_group.create_dataset('div', data=div)

    @staticmethod
    def read(file_path: str):
        """
        Read the search results from an HDF5 file and return the data, metadata and references.
        References contain the TOD ID, search profile index, and the shape of the right-hand side and normal matrix arrays.

        Static because the operation only needs the file path, not a configured
        Storage instance — lets callers do `Storage.read(path)` directly without
        constructing an output-path Storage they don't need.
        """
        references = []
        data = []
        metadata = {}

        with h5py.File(file_path, 'r') as f:
            metadata = read_hdf5_metadata(f.attrs)

            for tod_id in f.keys():
                tod_group = f[tod_id]

                rhs = tod_group['rhs'][:]
                div = tod_group['div'][:]

                references.append({
                    'tod_id': tod_id,
                    'rhs_shape': rhs.shape,
                    'div_shape': div.shape,
                })
                data.append((tod_id, rhs, div))

        df = pd.DataFrame(references)
        df.sort_values(by=['tod_id'], inplace=True)
        references = df.reset_index(drop=True)

        # Define the structured dtype for the data array
        max_length_tod_id = references['tod_id'].str.len().max()
        rhs_shape = references.iloc[0]['rhs_shape']
        div_shape = references.iloc[0]['div_shape']
        dtype = [
            ('tod_id', f'U{max_length_tod_id}'),
            ('rhs', 'f4', rhs_shape),
            ('div', 'f4', div_shape)
        ]

        data = np.array(data, dtype=dtype)

        return data, metadata, references

    @classmethod
    def metadata_from_directory(cls, directory: str, extension: str = '.hdf5') -> pd.DataFrame:
        """
        Load search results metadata from a directory containing HDF5 files.

        Parameters:
        - directory (str): The path to the directory containing the search results files.
        - extension (str): The file extension to filter the files by. Default is '.hdf5'.

        Returns:
        - pd.DataFrame: A DataFrame containing the metadata from the search results. The first column is the file path.
        """

        attributes = ['date', 'id', 'title', 'target', 'scenario_seed', 'search_profiles',
                      'operations', 'polarization_components', 'comments']
        data = {attr: [] for attr in attributes}
        data['file'] = []
        data['tods'] = []

        for filename in sorted(os.listdir(directory)):
            if filename.endswith(extension):
                filepath = os.path.join(directory, filename)
                data['file'].append(filepath)

                with h5py.File(filepath, 'r') as f:
                    metadata = read_hdf5_metadata(f.attrs)
                    for key in attributes:
                        if key not in metadata:
                            metadata[key] = ''
                        data[key].append(metadata[key])

                    data['tods'].append(len(list(f.keys())))

        df = pd.DataFrame(data)
        return df

def read_hdf5_metadata(attrs) -> dict:
    """
    Read the metadata from an HDF5 file and return them as a dictionary.
    """

    metadata = {}

    for key in attrs:
        value = attrs.get(key, None)
        if isinstance(value, str): 
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass

        if isinstance(value, np.ndarray):
            metadata[key] = tuple(value)
        elif isinstance(value, str) and value == 'None':
            metadata[key] = None
        else:
            metadata[key] = value

    return metadata

def count_files_with_extension(directory: str, extension: str = '.hdf5') -> int:
    """
    Count the number of files with a given extension in a directory.

    Parameters:
    directory (str): The path to the directory.
    extension (str): The file extension to filter the files by. Default is '.hdf5'.

    Returns:
    int: The number of files with the given extension.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"The path {directory} is not a valid directory.")
    
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            count += 1
    
    return count