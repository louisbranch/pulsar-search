import pytest
import numpy as np
from pulsar import RemovalProfile, ConstantProfile, FilterOptions, MultiplicativeGaussianNoise
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

    def test_seed_is_stable_for_identical_params(self, target):
        a = Scenario(title='a', target=target, search_profiles=[ConstantProfile(target=target)])
        b = Scenario(title='b-different', target=target, search_profiles=[ConstantProfile(target=target)])
        # title and metadata are stripped before hashing — same seed expected.
        assert a.seed == b.seed

    def test_seed_changes_when_filter_changes(self, target):
        a = Scenario(title='a', target=target,
                     search_profiles=[ConstantProfile(target=target)],
                     filter=FilterOptions(method='butterworth', cutoff=1.0))
        b = Scenario(title='a', target=target,
                     search_profiles=[ConstantProfile(target=target)],
                     filter=FilterOptions(method='fft', fknee=3.0))
        assert a.seed != b.seed

    def test_seed_changes_when_noise_changes(self, target):
        a = Scenario(title='a', target=target,
                     search_profiles=[ConstantProfile(target=target)],
                     noise=MultiplicativeGaussianNoise(mean=1.0, std=0.1))
        b = Scenario(title='a', target=target,
                     search_profiles=[ConstantProfile(target=target)],
                     noise=MultiplicativeGaussianNoise(mean=1.0, std=0.5))
        assert a.seed != b.seed

    def test_str_reports_profile_and_operation_counts(self, target, signal_profile):
        s = Scenario(
            title='probe',
            target=target,
            search_profiles=[signal_profile, signal_profile, signal_profile],
            operations=[SignalOperation(profile=signal_profile)],
        )
        text = str(s)
        assert 'probe' in text
        assert '3 target profiles' in text
        assert '1 operations' in text

    def test_str_with_no_operations(self, target, signal_profile):
        s = Scenario(title='x', target=target, search_profiles=[signal_profile])
        assert '0 operations' in str(s)

    def test_as_dict_contains_all_fields(self, target, signal_profile):
        s = Scenario(title='x', target=target, search_profiles=[signal_profile])
        d = s.as_dict()
        for key in ('title', 'target', 'search_profiles', 'operations',
                    'calibration_options', 'filter', 'filter_sources',
                    'noise', 'polarization_components', 'config', 'metadata',
                    'seed'):
            assert key in d

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

    def test_amplitude_short_tuple_pads_with_zero(self, signal_profile):
        # Tuple shorter than ncomp: missing components are zero.
        op = SignalOperation(signal_profile, amp=(1.0, 2.0))
        np.testing.assert_allclose(op.amplitude(3), np.array([[1.0, 2.0, 0.0]]))

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