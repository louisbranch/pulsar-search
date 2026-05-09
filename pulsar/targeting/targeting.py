from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from typing import List, Optional, Tuple
import h5py
import os

import numpy as np
from mpi4py import MPI

from ..config import config
from ..instrument import Instrument
from .. import log
from ..pointing_models import PointingModel
from ..target import Target
from ..tod import TOD

# Define constants for MPI tags
SUPERVISOR = 0
WAITING = 1
RESULT = 2
STOP = None

@dataclass
class Targeting:
    """
    Pre-compute, for each TOD, which sample ranges fall within each target's
    avoidance radius. The result is stored to a single HDF5 file keyed by
    `tod_id` and target hash, so a later search (or analysis) can quickly
    decide which TODs see a given target without re-running the full pointing
    pipeline.

    Runs in parallel via MPI; the supervisor distributes TODs to workers and
    appends their results to the HDF5 file.
    """

    targets: List[Target]
    tod_path: str
    output_path: str
    tod_limit: int = 0
    started_at: datetime = datetime.now()
    ended_at: Optional[datetime] = None
    instrument: Optional[Instrument] = None
    pointing_model: Optional[PointingModel] = None

    def __post_init__(self):
        if self.instrument is None:
            self.instrument = config.instrument
        
        if self.pointing_model is None:
            self.pointing_model = config.instrument.pointing_model()

    def run(self):
        """
        Run targeting in parallel via MPI. Rank 0 acts as supervisor; the
        remaining ranks process one TOD at a time until exhausted.
        """
        if MPI is None:
            log.critical('MPI is not installed. Please install mpi4py to use parallel processing.')

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            self._supervisor(comm)
        else:
            self._worker(comm)

    def _supervisor(self, comm):
        """
        The main process acts as a supervisor, distributing the work to the remaining processes.
        It keeps track of the active workers and sends the next TOD to process to each worker.
        Once all TOD ids have been processed, the supervisor sends a STOP signal to each worker.
        """

        tod_ids = self.instrument.tod_ids(self.tod_path)
        tod_ids = tod_ids[:self.tod_limit] if self.tod_limit > 0 else tod_ids
        log.debug(f'Found {len(tod_ids)} TODs to process.')

        target_ids = [target_hash(target) for target in self.targets]
        missing = read_hdf5_file(self.output_path, tod_ids, target_ids)
        ids = list(missing.keys())
        log.debug(f'Found {len(ids)} TODs with missing targets.')

        size = comm.Get_size()
        active_workers = 0

        def end_process():
            # Ensure all remaining workers receive the STOP signal
            log.info('Stopping all workers.')
            for worker_id in range(1, size):
                comm.send(STOP, dest=worker_id, tag=WAITING)
            MPI.Finalize()

        if len(missing) == 0:
            log.info('No TODs with missing targets to process. Exiting.')
            end_process()
            return

        def targets(id):
            target_ids = missing[id]
            targets = []
            for target_id in target_ids:
                index = target_ids.index(target_id)
                targets.append(self.targets[index])
            return targets

        def send_job(ids, worker_id):
            id = ids[0]
            comm.send((id, targets(id)), dest=worker_id, tag=WAITING)
            return ids[1:]

        # Distribute the first batch of work
        for worker_id in range(1, size):
            if len(ids) > 0:
                ids = send_job(ids, worker_id)
                active_workers += 1

        while active_workers > 0:
            response = comm.recv(source=MPI.ANY_SOURCE, tag=RESULT)
            tod_id, samples = response['results']

            if len(ids) > 0:
                ids = send_job(ids, response['worker'])
            else:
                comm.send(STOP, dest=response['worker'], tag=WAITING)  # Signal worker to stop
                active_workers -= 1

            append_to_hdf5_file(self.output_path, tod_id, targets(tod_id), samples)

        self.ended_at = datetime.now()
        elapsed_time = self.ended_at - self.started_at

        end_process()

        log.info(f'Parallel search completed in {elapsed_time}.')

    def _worker(self, comm):
        """
        Worker process that receives a TOD and a list of scenarios from the supervisor.
        It processes the TOD for each scenario and sends the result back to the supervisor.
        The worker will continue to process TODs until it receives a STOP signal from the supervisor.
        """
        rank = comm.Get_rank()
        while True:
            value = comm.recv(source=SUPERVISOR, tag=WAITING)
            if value is STOP:
                break

            id, targets = value
            tod = next(self.instrument.tods(self.tod_path, ids=[id]))
            results = self._search_targets(tod, targets)

            comm.send({'worker': rank, 'results': results}, dest=SUPERVISOR, tag=RESULT)

    def _search_targets(self, tod: TOD, targets: List[Target]):
        """
        For a single TOD, return the sample ranges that fall within each
        target's avoidance radius. Calibration is run first so coordinates
        align with the calibrated scan metadata.
        """
        log.debug(f'Searching for targets in TOD {tod.id}...')
        tod.calibrate()
        results = []
        for target in targets:
            log.debug(f'Searching for target {target.name}...')
            _, samples = tod.locate_source(target.ra, target.dec, target.radius)
            ranges = samples.ranges
            log.debug(f'Found {len(ranges)} samples for target {target.name}.')
            results.append(ranges)
        return tod.id, results

def target_hash(target):
    return hashlib.md5(f"{target.ra}{target.dec}{target.radius}".encode()).hexdigest()

def read_hdf5_file(file_path, tod_ids, target_ids):
    missing_tods = {}

    if not os.path.exists(file_path):
        # If the file doesn't exist, return all TODs with all target IDs as missing
        for tod_id in tod_ids:
            missing_tods[tod_id] = target_ids
        return missing_tods
    
    # Open the file in read mode
    with h5py.File(file_path, 'r') as f:
        tods_group = f.get('/tods', None)
        targets_group = f.get('/targets', None)

        if tods_group is None or targets_group is None:
            # If either the TOD or Targets group doesn't exist, consider all missing
            for tod_id in tod_ids:
                missing_tods[tod_id] = target_ids
            return missing_tods

        for tod_id in tod_ids:
            # Check if the TOD exists under /tods
            if tod_id not in tods_group:
                missing_tods[tod_id] = target_ids
            else:
                missing_targets = []
                for target_id in target_ids:
                    # Check if each target exists under the specific TOD group
                    if target_id not in tods_group[tod_id]:
                        missing_targets.append(target_id)
                if missing_targets:
                    missing_tods[tod_id] = missing_targets

    return missing_tods

def append_to_hdf5_file(file_path, tod_id, targets, samples_ranges):
    with h5py.File(file_path, 'a') as f:
        # Access or create /targets group
        if 'targets' not in f:
            targets_group = f.create_group('/targets')
        else:
            targets_group = f['/targets']

        # Access or create /tods group
        if 'tods' not in f:
            tods_group = f.create_group('/tods')
        else:
            tods_group = f['/tods']

        # Ensure the /tods/tod_id exists
        if tod_id not in tods_group:
            tod_group = tods_group.create_group(tod_id)
        else:
            tod_group = tods_group[tod_id]

        # Add or update target information in /targets and /tods/tod_id
        for i, target in enumerate(targets):
            target_id = target_hash(target)

            # Step 1: Store target attributes globally under /targets
            if target_id not in targets_group:
                target_group = targets_group.create_group(target_id)
                target_group.attrs['name'] = target.name
                target_group.attrs['ra'] = target.ra
                target_group.attrs['dec'] = target.dec
                target_group.attrs['radius'] = target.radius

            # Step 2: Store target-specific data under /tods/tod_id
            if target_id not in tod_group:
                target_group = tod_group.create_group(target_id)
                target_group.create_dataset('samples_range', data=samples_ranges[i])
