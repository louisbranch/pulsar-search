from datetime import datetime as dt
import numpy as np
import pytest
from pulsar.search.result import Result
from .fixtures import *

class TestResult:

    def test_filename(self, scenario):
        now = dt.strptime('2021-01-01', '%Y-%m-%d')
        scenario.seed = 1234
        scenario.target.name = 'target'
        result = Result(now, 'tod1', scenario, [], [])
        assert result.filename == 'target_1234.hdf5'

    def test_date(self):
        now = dt.strptime('2021-01-01', '%Y-%m-%d')
        result = Result(now, 'tod1', None, [], [])
        assert result.date == '2021-01-01'

    def test_metadata(self, scenario):
        scenario.metadata = {'title': 'new_title', 'custom': 'value'}
        result = Result(dt.now(), 'tod1', scenario, [], [])
        metadata = result.metadata

        assert metadata['title'] == 'new_title'
        assert metadata['date'] == result.date
        assert metadata['scenario_seed'] == scenario.seed
        assert metadata['target'] == scenario.target.as_dict()
        assert metadata['search_profiles'] == [profile.as_dict() for profile in scenario.search_profiles]
        assert metadata['operations'] == [operation.as_dict() for operation in scenario.operations]
        assert metadata['filter'] == scenario.filter.as_dict()
        assert metadata['noise'] == scenario.noise.as_dict()
        assert metadata['polarization_components'] == scenario.polarization_components.value
        assert metadata['custom'] == 'value'
