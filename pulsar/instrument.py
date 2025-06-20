from typing import Generator, List, Optional, Protocol

import numpy as np

from .tod import TOD
from .pointing_models import PointingModel

class Instrument(Protocol): # pragma: no cover
    """
    Protocol to represent a generic astronomical instrument capable of generating 
    Time-Ordered Data (TOD) objects from observations.

    This protocol defines the standard interface for interacting with instruments
    to retrieve observational data processed into TOD format.
    """

    def tods(self, path: str, limit: int = 0, dets: Optional[List[int]] = None,
             ids: Optional[List[str]] = None) -> Generator[TOD, None, None]: # pragma: no cover
        """
        Streams TOD objects based on a given path.

        Parameters:
            path (str): The query or path specifying where to find the data or how to query it.
            limit (int, optional): The maximum number of TOD objects to retrieve. If zero, no limit is applied.
            dets (List[int], optional): A list of detector indices to filter the TODs by specific detectors.
            ids (List[str], optional): A list of TOD IDs to filter the TODs by specific IDs.

        Yields:
            Generator[TOD, None, None]: A generator of TOD objects fulfilling the specified criteria.
        """
        pass

    def tod_ids(self, path: str) -> List[str]: # pragma: no cover
        """
        Retrieves a list of TOD IDs based on a given path.

        Parameters:
            path (str): The query or path specifying where to find the data or how to query it.

        Returns:
            List[str]: A list of TOD IDs fulfilling the specified criteria.
        """
        pass

    def pointing_model(self, *args, **kwargs) -> PointingModel: # pragma: no cover
        """
        Returns the pointing model object for the instrument.
        This object maps sky signals into TOD.
        """
        pass

    def create_map(self, *args, **kwargs) -> np.ndarray: # pragma: no cover
        """
        Generates a map from the instrument's TOD.
        """
        pass

    def plot_map(self, *args, **kwargs) -> None: # pragma: no cover
        """
        Displays a map generated from the instrument's TOD.
        """
        pass
