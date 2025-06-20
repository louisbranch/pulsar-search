from copy import copy
import numpy as np
import numpy.testing as npt
import pytest
from pulsar import RemovalProfile, create_boxcar_profiles, create_von_mises_profiles
from .fixtures import *

class TestConstantProfile:

    def test_profile(self, constant_profile):
        phases = [0.1, 0.2, 0.3]
        expected_profile = [1.0, 1.0, 1.0]
        npt.assert_allclose(constant_profile.profile(phases), expected_profile)

    def test_as_dict(self, constant_profile):
        expected_dict = {
            'name': 'Constant',
            'target': constant_profile.target.as_dict(),
            'timing': 'PeriodTimingModel',
        }
        assert constant_profile.as_dict() == expected_dict

    def test_obstime2phase(self, constant_profile):
        ctime = 0.5
        expected_phase = 0.707
        assert pytest.approx(constant_profile.obstime2phase(ctime), 0.001) == expected_phase
        constant_profile.target.phase_sign = '-'
        expected_phase = 0.693
        assert pytest.approx(constant_profile.obstime2phase(ctime), 0.001) == expected_phase

    def test_obstime2profile(self, constant_profile):
        ctime = 0.5
        expected_profile = 1.0
        assert pytest.approx(constant_profile.obstime2profile(ctime), 0.001) == expected_profile

    def test_obstime2flux(self, constant_profile):
        ctime = 0.5
        expected_flux = 1.0 * constant_profile.target.amp
        assert pytest.approx(constant_profile.obstime2flux(ctime), 0.001) == expected_flux

    def test_src(self, constant_profile):
        expected_src = np.array([0.5, 0.3, 10.0], dtype=float)
        npt.assert_allclose(constant_profile.src, expected_src)

        constant_profile.target = None
        assert constant_profile.src is None

    def test_str(self, constant_profile):
        expected_str = 'Signal Profile: Constant, ' + str(constant_profile.target)
        assert str(constant_profile) == expected_str

class TestVonMisesProfile:

    def test_profile(self, von_mises_profile):
        phases = np.array([0.1, 0.2, 0.3])
        expected_profile = [79.547, 0.0668, 0.001]
        npt.assert_allclose(von_mises_profile.profile(phases), expected_profile, atol=1e-2)

    def test_obstime2profile(self, von_mises_profile):
        ctime = 0.5
        expected_profile = 1.996e-5
        assert pytest.approx(von_mises_profile.obstime2profile(ctime), abs=1e-6) == expected_profile

    def test_obstime2flux(self, von_mises_profile):
        ctime = 0.5
        expected_flux = 1.996e-5 * von_mises_profile.target.amp
        assert pytest.approx(von_mises_profile.obstime2flux(ctime), abs=1e-6) == expected_flux

    def test_as_dict(self, von_mises_profile):
        expected_dict = {
            'name': 'Von Mises',
            'mu': 0.5,
            'target': von_mises_profile.target.as_dict(),
            'timing': 'PeriodTimingModel',
        }
        assert von_mises_profile.as_dict() == expected_dict

class TestBoxcarProfile:

    def test_profile(self, boxcar_profile):
        phases = [0.1, 0.2, 0.3, 0.4]
        expected_profile = [0, 0, 0, 1]
        npt.assert_allclose(boxcar_profile.profile(phases), expected_profile)

    def test_obstime2profile(self, boxcar_profile):
        ctime = 0.5
        expected_profile = 0.0
        assert pytest.approx(boxcar_profile.obstime2profile(ctime), 0.001) == expected_profile
        ctime = 0.75
        expected_profile = 1.0
        assert pytest.approx(boxcar_profile.obstime2profile(ctime), 0.001) == expected_profile

    def test_obstime2flux(self, boxcar_profile):
        ctime = 0.5
        expected_flux = 0.0
        assert pytest.approx(boxcar_profile.obstime2flux(ctime), 0.001) == expected_flux
        ctime = 0.75
        expected_flux = 1.0 * boxcar_profile.target.amp
        assert pytest.approx(boxcar_profile.obstime2flux(ctime), 0.001) == expected_flux

    def test_as_dict(self, boxcar_profile):
        expected_dict = {
            'name': 'Boxcar',
            'num_bins': 4,
            'bin_index': 2,
            'target': boxcar_profile.target.as_dict(),
            'timing': 'PeriodTimingModel',
        }
        assert boxcar_profile.as_dict() == expected_dict

class TestRemovalProfile:

    def test_as_dict(self, target):
        removal = RemovalProfile(target=target)
        expected_dict = {
            'name': 'Removal',
            'target': removal.target.as_dict(),
            'timing': 'PeriodTimingModel',
        }
        assert removal.as_dict() == expected_dict

    def test_profile(self, target):
        removal = RemovalProfile(target=target)
        phases = [0.1, 0.2, 0.3]

        assert np.array_equal(removal.profile(phases), np.zeros_like(phases))

def test_create_boxcar_profiles(target):
    profiles = create_boxcar_profiles(target=target, num_bins=4)
    assert len(profiles) == 4
    assert all(isinstance(profile, BoxcarProfile) for profile in profiles)
    assert all(profile.num_bins == 4 for profile in profiles)
    assert all(profile.bin_index == i for i, profile in enumerate(profiles))

def test_create_von_mises_profile(target):
    profiles = create_von_mises_profiles(target=target, n=4)
    assert len(profiles) == 4
    assert all(isinstance(profile, VonMisesProfile) for profile in profiles)
    assert all(profile.target.phi0 == i * 0.25 for i, profile in enumerate(profiles))