from copy import copy, deepcopy
from dataclasses import dataclass, field
from datetime import datetime
import traceback
from typing import List, Optional, Tuple
import os

import numpy as np
from mpi4py import MPI

from ..config import config
from ..filters import filter
from ..flux_estimator import FluxEstimator
from ..instrument import Instrument
from .. import log
from ..pointing_models import PointingModel
from ..tod import TOD
from .extension import ExtensionManager, Extension
from .result import Result
from .scenario import Scenario
from .storage import Storage

# Define constants for MPI tags
SUPERVISOR = 0
WAITING = 1
RESULT = 2
STOP = None

@dataclass
class Search:
    """
    Search performs a search on the TOD files with the given scenarios.

    Attributes:
    scenarios (List[Scenario]): List of scenarios to search for.
    tod_path (str): Path to the TOD files.
    output_path (str): Path to save the search results.
    tod_ids (List[str]): List of TOD IDs to search. Default is None (all TODs).
    tod_limit (int): Limit the number of TODs to process. Default is 0 (no limit).
    parallel (bool): Use parallel processing with MPI. Default is True.
    started_at (datetime): The start time of the search. Default is the current time.
    ended_at (datetime): The end time of the search. Default is None.
    instrument (Instrument): The instrument to use for the search. Default is the instrument from the config.
    pointing_model (PointingModel): The pointing model to use for the search. Default is the pointing model from the instrument.
    save (bool): Save the results to disk. Default is True.
    batch_size (int): Number of scenarios to process in a single batch. Default is None (no batching).
    extensions (List[Extension]): List of extensions to use during the search. Default is an empty list.
    """

    scenarios: List[Scenario]
    tod_path: str
    output_path: str
    tod_ids: Optional[List[str]] = None
    tod_limit: int = 0
    parallel: bool = True
    started_at: datetime = datetime.now()
    ended_at: Optional[datetime] = None
    instrument: Optional[Instrument] = None
    pointing_model: Optional[PointingModel] = None
    save: bool = True
    batch_size: Optional[int] = None
    extensions: List[Extension] = field(default_factory=list)
    _extension_manager: ExtensionManager = ExtensionManager()

    def __post_init__(self):
        # FIXME: there is a race condition here if the output path is created by another process
        # before the search starts. This should be fixed by the storage object.
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        if self.instrument is None:
            self.instrument = config.instrument
        
        if self.pointing_model is None:
            self.pointing_model = config.instrument.pointing_model()

        for extension in self.extensions:
            self._extension_manager.register_extension(extension)
    
    def run(self):
        """
        Run the search on the TOD files with the given scenarios.
        """
        if self.parallel:
            self.run_parallel()
        else:
            self.run_sequential()

    def run_parallel(self):
        """
        Perform a parallel search using MPI. The main process acts as a supervisor, distributing
        the work to the remaining processes. Each worker processes a single TOD and sends the results
        back to the supervisor to be saved to disk.
        """
        if MPI is None:
            log.critical('MPI is not installed. Please install mpi4py to use parallel processing.')

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            self._supervisor(comm)
        else:
            self._worker(comm)

    def run_sequential(self):
        """
        Perform a sequential search for each TOD in the TOD path with the given scenarios. 
        The result of each scenario is saved to disk as a HDF5 file.
        """
        
        ids, scenarios, storage = self._setup_search()

        if len(scenarios) == 0:
            log.info('No scenarios to process. Exiting.')
            return

        for original_tod in self.instrument.tods(self.tod_path, ids=ids):
            results = self._search_scenarios(original_tod, scenarios)
            storage.update(results)

        self.ended_at = datetime.now()
        elapsed_time = self.ended_at - self.started_at

        log.info(f'Sequential search completed in {elapsed_time}.')

    def _setup_search(self) -> Tuple[List[str], List[Scenario], Storage]:
        """
        Get the list of TOD IDs to search and create a storage object to save the results.
        If the search is resuming, the missing TOD IDs are returned instead.
        If the save flag is False, a null storage object is returned to discard the results.

        Find the missing TOD IDs given the scenarios and return the list of TOD IDs and scenarios to process.
        """

        if self.tod_ids is None:
            ids = self.instrument.tod_ids(self.tod_path)
        else:
            ids = self.tod_ids

        ids = ids[:self.tod_limit] if self.tod_limit > 0 else ids
        storage = Storage(self.output_path)
        if not self.save:
            log.info(f'Starting search with {len(self.scenarios)} scenarios without saving results to disk. Processing {len(ids)} TOD IDs')
            null_storage = Storage('/dev/null')
            return ids, list(self.scenarios), null_storage

        scenarios, missing_ids, completed = storage.missing_ids(ids, self.scenarios)
        if len(completed) > 0:
            log.info(f'Skipping {len(completed)} completed scenarios')
        if len(scenarios) > 0:
            log.info(f'Starting search with {len(scenarios)} scenarios. Processing {len(missing_ids)} TOD IDs')

        return missing_ids, scenarios, storage

    def _supervisor(self, comm):
        """
        The main process acts as a supervisor, distributing the work to the remaining processes.
        It keeps track of the active workers and sends the next TOD to process to each worker.
        Once all TOD ids have been processed, the supervisor sends a STOP signal to each worker.
        """

        ids, scenarios, storage = self._setup_search()
        size = comm.Get_size()
        active_workers = 0

        def end_process():
            # Ensure all remaining workers receive the STOP signal
            log.info('Stopping all workers.')
            for worker_id in range(1, size):
                comm.send(STOP, dest=worker_id, tag=WAITING)
            MPI.Finalize()

        if len(scenarios) == 0:
            log.info('No scenarios to process. Exiting.')
            end_process()
            return
        elif len(scenarios) == 1:
            log.info(f'Starting search for {scenarios[0].title} scenario')

        current_batch_index = 0

        # Split scenarios into batches if batch_size is specified
        if self.batch_size:
            batches = [scenarios[i:i + self.batch_size] for i in range(0, len(scenarios), self.batch_size)]
            log.info(f'Splitting scenarios into {len(batches)} batches of size {self.batch_size}')
            log.info(f'Starting batch {current_batch_index + 1}/{len(batches)}')
        else:
            batches = [scenarios]

        current_batch = batches[current_batch_index]
        batch_ids = ids.copy()
        
        # Distribute the first batch of work
        for worker_id in range(1, size):
            if len(batch_ids) > 0:
                comm.send((batch_ids[0], current_batch), dest=worker_id, tag=WAITING)
                batch_ids = batch_ids[1:]
                active_workers += 1

        while active_workers > 0:
            response = comm.recv(source=MPI.ANY_SOURCE, tag=RESULT)
            results = response['results']
            storage.update(results)

            if len(batch_ids) > 0:
                comm.send((batch_ids[0], current_batch), dest=response['worker'], tag=WAITING)
                batch_ids = batch_ids[1:]
            else:
                current_batch_index += 1
                if current_batch_index < len(batches):
                    log.info(f'Starting batch {current_batch_index + 1}/{len(batches)}')
                    current_batch = batches[current_batch_index]
                    batch_ids = ids.copy()
                    comm.send((batch_ids[0], current_batch), dest=response['worker'], tag=WAITING)
                    batch_ids = batch_ids[1:]
                else:
                    comm.send(STOP, dest=response['worker'], tag=WAITING)  # Signal worker to stop
                    active_workers -= 1


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

            id, scenarios = value
            original_tod = next(self.instrument.tods(self.tod_path, ids=[id]))
            results = self._search_scenarios(original_tod, scenarios)

            comm.send({'worker': rank, 'results': results}, dest=SUPERVISOR, tag=RESULT)

    def _search_scenarios(self, original_tod: TOD,
                          scenarios: List[Scenario]) -> List[Result]:
        """
        For a given TOD, process the scenario and return the results.
        Each scenario deep copies the original TOD and applies the scenario parameters.
        """

        results : List[Result] = []

        for scenario in scenarios:

            seed = scenario.seed
            np.random.seed(seed)

            log.debug(f'Searching scenario {scenario}')

            if len(scenarios) == 1:
                tod = original_tod
            else:
                tod = deepcopy(original_tod)

            @self._extension_manager.wrap_step('calibrate', None, original_tod)
            def calibrate(tod):
                if scenario.calibration_options is not None:
                    log.debug(f'Calibrating TOD {tod.id} with options: {scenario.calibration_options}')
                options = scenario.calibration_options or {}
                tod.calibrate(**options)

            calibrate(tod)

            try: 
                rhs, div = self._run_search(tod, scenario)
            except Exception as e:
                log.error(f'Error processing TOD {tod.id} with scenario {scenario}: {e}')
                traceback.print_exc()
                continue

            result = Result(self.started_at, tod.id, scenario, rhs, div)
            results.append(result)

        return results

    def _run_search(self, tod: TOD, scenario: Scenario) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the search for a given scenario on the TOD.

        Parameters:
        tod (TOD): The Time-Ordered Data to search in.
        scenario (Scenario): The scenario to search for.

        Returns:
        Tuple[np.ndarray, np.ndarray]: The right-hand side and divergence of the flux calculation.
        """

        target = scenario.target
        operations = scenario.operations or []
        ncomp = scenario.polarization_components.value

        config = {}
        if scenario.config is not None:
            config = scenario.config
            log.debug(f'Applying configuration: {config} to scenario {scenario}')

        # Set the pointing model for the TOD
        pmodel = self.pointing_model

        @self._extension_manager.wrap_step('operation', scenario, tod)
        def apply_operation(operation, tod):
            if operation.profile.target is None:
                operation.profile.target = copy(target)

            if operation.is_removal:
                tod.remove_source(*operation.position)
            else:
                pconfig = config.get('pointing_model', {})
                pmat = pmodel(tod, [operation.profile], **pconfig)
                pmat.forward(tod.data, operation.amplitude(ncomp))

        @self._extension_manager.wrap_step('noise', scenario, tod)
        def apply_noise(noise, tod):
            noise.apply(tod.data)

        @self._extension_manager.wrap_step('filter', scenario, tod)
        def apply_filter(tod, options, sources):
            filter(tod, options, sources)

        @self._extension_manager.wrap_step('flux', scenario, tod)
        def estimate_flux(tod, search_profiles, ncomp):
            for profile in search_profiles:
                if profile.target is None:
                    profile.target = copy(target)
            flux = FluxEstimator(pmodel)

            profile = search_profiles[0]

            if profile.phase_exclusive:
                algorithm = flux.estimate_individual
                log.debug('Using flux estimation phase exclusive algorithm')
            else:
                algorithm = flux.estimate
                log.debug('Using flux estimation phase inclusive algorithm')

            return algorithm(tod, search_profiles, ncomp)

        if scenario.filter is not None:
            apply_filter(tod, scenario.filter, scenario.filter_sources)

        [apply_operation(operation, tod) for operation in operations]
        if scenario.noise is not None:
            apply_noise(scenario.noise, tod)

        return estimate_flux(tod, scenario.search_profiles, ncomp)