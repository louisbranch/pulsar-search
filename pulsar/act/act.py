from typing import List, Optional, Generator

from enlib import config

from .io import tods, tod_ids
from .tod import TOD
from .pointing_models import PmatTotTransient
from .map import create_map, plot_map
from .. import log

class ACT:
    """
    A concrete implementation of an astronomical instrument, specifically for the Atacama Cosmology Telescope (ACT).

    This class provides methods to access and manipulate observational data specific to ACT.
    """

    def __init__(self, dataset: str = 'dr6v4', root: Optional[str] = None,
                 pointing_file: Optional[str] = None, verbose: bool = False) -> None:
        """
        Initializes the ACT instrument with a specific dataset and verbosity setting.

        Parameters:
            dataset (str): The dataset version to use for queries and data retrieval.
            root (str, optional): The root directory where the data is stored.
            pointing_file (str, optional): The path to a file containing pointing corrections.
            verbose (bool): If True, enables verbose output during operations.
        """
        self.verbose = verbose
        config.set('dataset', dataset)

        if root is not None:
            log.debug(f'Setting ACT root directory to: {root}')
            config.set('root', root)

        if pointing_file is not None:
            log.debug(f'Using pointing file: {pointing_file}')
            config.set('file_override', f'@{pointing_file}')

    def tods(self, query: str, limit: int = 0, dets: Optional[List[int]] = None,
             ids: Optional[List[str]] = None) -> Generator[TOD, None, None]:
        """
        Retrieves a list of TOD objects based on a specific query.

        Parameters:
            query (str): The query specifying selection criteria for data retrieval.
            limit (int, optional): The maximum number of TOD objects to retrieve. If zero, no limit is applied.
            dets (List[int], optional): A list of detector indices to filter the TOD files by specific detectors.
            ids (List[str], optional): A list of TOD IDs to filter the TOD files by specific IDs.

        Yields:
            Generator[TOD, None, None]: A generator of TOD objects fulfilling the specified criteria.
        """
        return tods(query, limit=limit, dets=dets, ids=ids, verbose=self.verbose)

    def tod_ids(self, query: str) -> List[str]:
        """
        Retrieves a list of TOD IDs based on a specific query.

        Parameters:
            query (str): The query specifying selection criteria for data retrieval.

        Returns:
            List[str]: A list of TOD IDs fulfilling the specified criteria.
        """
        return tod_ids(query)

    def pointing_model(self, *args, **kwargs) -> PmatTotTransient:
        """
        Returns the pointing model object for the instrument.
        This object maps sky signals into TOD.
        """
        return PmatTotTransient

    def create_map(self, *args, **kwargs) -> None:
        """
        Generates a map from the instrument's TOD data.
        """
        
        return create_map(*args, **kwargs)

    def plot_map(self, *args, **kwargs) -> None:
        """
        Displays a map generated from the instrument's TOD data.
        """
        return plot_map(*args, **kwargs)