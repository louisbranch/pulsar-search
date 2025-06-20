import pytest
from copy import copy

from pulsar import Target, ConstantProfile, VonMisesProfile, BoxcarProfile

@pytest.fixture
def target():
    return Target(name='crab', ra=0.5, dec=0.3, amp=10.0, T=0.335, D=0.1, phi0=0.2, t0=1.0, phase_sign='+')

@pytest.fixture
def constant_profile(target):
    return ConstantProfile(target=copy(target))

@pytest.fixture
def von_mises_profile(target):
    return VonMisesProfile(target=copy(target), mu=0.5)

@pytest.fixture
def boxcar_profile(target):
    return BoxcarProfile(target=copy(target), num_bins=4, bin_index=2)
