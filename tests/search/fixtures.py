import pytest

from pulsar import Target, ConstantProfile, FilterOptions, MultiplicativeGaussianNoise
from pulsar.search import Scenario, SignalOperation

@pytest.fixture
def target():
    return Target(name='Target', T=1.0, D=0.5, phi0=0.0, amp=1.0, ra=1.0, dec=1.5, t0=0.0, phase_sign='-')

@pytest.fixture
def signal_profile(target):
    return ConstantProfile(target=target)

@pytest.fixture
def signal_operation(signal_profile):
    return SignalOperation(profile=signal_profile)

@pytest.fixture
def scenario(target, signal_operation):
    return Scenario(
        title='Title',
        target=target,
        seed=1,
        search_profiles=[ConstantProfile()],
        operations=[signal_operation],
        filter=FilterOptions(),
        noise=MultiplicativeGaussianNoise(),
    )

@pytest.fixture
def scenarios(target):
    return [
        Scenario(title='Title1', target=target, seed=1, search_profiles=[ConstantProfile()]),
        Scenario(title='Title2', target=target, seed=2, search_profiles=[ConstantProfile()]),
    ]