from dataclasses import dataclass
from datetime import datetime as dt

import numpy as np

from .scenario import Scenario
from .. import log

@dataclass
class Result:
    """
    Store the results of a scenario for a single TOD.
    The result is a temporary data structure that is used to save the results to an HDF5 file.
    When using parallel processing, each worker will generate a list of these results for a single TOD,
    which are then saved to disk by the main process.
    """
    datetime: dt
    tod_id: str
    scenario: Scenario
    rhs: np.ndarray
    div: np.ndarray

    @property
    def filename(self, extension: str = '.hdf5') -> str:
        return f'{self.scenario.target.name}_{self.scenario.seed}{extension}'

    @property
    def date(self) -> str:
        return self.datetime.strftime('%Y-%m-%d')

    @property
    def metadata(self) -> dict:
        scenario = self.scenario

        target = scenario.target.as_dict()
        profiles = [profile.as_dict() for profile in scenario.search_profiles]
        operations = [operation.as_dict() for operation in scenario.operations] if scenario.operations is not None else None
        filter = scenario.filter.as_dict() if scenario.filter is not None else None
        noise = scenario.noise.as_dict() if scenario.noise is not None else None
        pol = scenario.polarization_components.value
        config = scenario.config if scenario.config is not None else None

        mapping = {
            'title': scenario.title,
            'date': self.date,
            'scenario_seed': scenario.seed,
            'target': target,
            'search_profiles': profiles,
            'operations': operations,
            'filter': filter,
            'noise': noise,
            'polarization_components': pol,
            'config': config,
        }

        if scenario.metadata is not None:
            for key, value in scenario.metadata.items():
                if key in mapping:
                    log.warning(f'Overwriting metadata key "{key}" with value "{value}" for scenario "{scenario.title}".')
                mapping[key] = value

        return mapping