from typing import Generator, List, Optional

from .tod import TOD
from .config import config

def tods(path: str, limit: int=0, dets: Optional[List[int]] = None,
         ids: Optional[List[str]] = None) -> Generator[TOD, None, None]:
    """
    Streams TOD objects based on a given path.
    See Also: Instrument.tods
    """
    return config.instrument.tods(path, limit=limit, dets=dets, ids=ids)

def tod_ids(path: str) -> List[str]:
    """
    Retrieves a list of TOD IDs based on a given path.
    See Also: Instrument.tod_ids
    """
    return config.instrument.tod_ids(path)
