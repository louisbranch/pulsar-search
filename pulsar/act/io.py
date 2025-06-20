from typing import List, Optional, Generator

from enact import actdata, actscan, filedb
from enlib import errors

from .tod import TOD
from .. import log

def tod_ids(query: str) -> List[str]:
    filedb.init()
    db = filedb.scans.select(query_path(query))
    return db.ids

def tods(query: str, limit: int = 0, dets: Optional[List[int]] = None,
         ids: Optional[List[str]] = None, verbose: bool = False) -> Generator[TOD, None, None]:

    filedb.init()
    db = filedb.scans.select(query_path(query))

    count = 0
    for id in db.ids:
        if ids is not None and id not in ids:
            continue

        log.debug(f"Processing TOD: {id}")
        entry = filedb.data[id]
        try:
            scan = actscan.ACTScan(entry, subdets=dets, verbose=verbose)
            if dets is None and scan.ndet < 2:
                raise errors.DataMissing('Not enough detectors')
            elif scan.nsamp < 1:
                raise errors.DataMissing('Not enough data samples')
        except errors.DataMissing as e:
            log.debug(f"Data missing for TOD ID: {id} - {e}")
            continue  # Skip this TOD and move to the next

        det_ids = scan.dets
        dataset = actdata.read(entry, dets=det_ids, verbose=verbose)

        band = entry['tag']
        ar = entry['id'][-3:]

        tod = TOD(id, scan, dataset, band, ar)

        count += 1
        yield tod

        if limit > 0 and count >= limit:
            log.debug(f"Reached limit of {limit} TODs")
            return

def query_path(query: str) -> str:
    if not query.startswith('@'):
        query = '@' + query
    return query