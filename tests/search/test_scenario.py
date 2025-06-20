import pytest
import numpy as np
from pulsar import RemovalProfile
from pulsar.search.scenario import Scenario, SignalOperation
from .fixtures import *

class TestScenario:

    def test_init(self, target, signal_profile):
        # missing required fields
        with pytest.raises(TypeError):
            Scenario()

        # missing search_profiles
        with pytest.raises(ValueError):
            Scenario(title='test', search_profiles=[], target=target)

        # wrong polarization_components
        with pytest.raises(ValueError):
            Scenario(title='test', search_profiles=[signal_profile], target=target, polarization_components='a')

        # generate seed
        s = Scenario(title='test', search_profiles=[signal_profile], target=target)
        assert s.seed is not None

class TestSignalOperation:

    def test_position(self, signal_profile):
        op = SignalOperation(signal_profile)
        assert op.position == (1.0, 1.5, 0)

    def test_amplitude(self, signal_profile):

        op = SignalOperation(signal_profile)

        # from target
        assert op.amplitude(1) == ((1.0))
        assert np.allclose(op.amplitude(3), np.array([1.0, 0, 0]))

        # from self
        op.amp = (1.0, 2.0, 3.0)
        assert op.amplitude(1) == ((1.0))
        assert np.allclose(op.amplitude(3), np.array([1.0, 2.0, 3.0]))

    def test_as_dict(self, signal_profile):
        op = SignalOperation(signal_profile, amp=(1.0, 2.0, 3.0))
        assert op.as_dict() == {
            'profile': signal_profile.as_dict(),
            'amp': (1.0, 2.0, 3.0)
        }

    def test_is_removal(self, signal_profile):
        op = SignalOperation(signal_profile)
        assert not op.is_removal

        op.profile = RemovalProfile()
        assert op.is_removal