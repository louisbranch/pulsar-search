import numpy.testing as npt
import pytest
from pulsar.polarization import calculate_stokes_parameters

@pytest.mark.parametrize("name, inputs, expected", [
    ('Crab in X-Ray', (10.0, 19.0, 145.5), (10.0, 0.680899, -1.773803))
])
def test_calculate_stokes_parameters(name, inputs, expected):
    result = calculate_stokes_parameters(*inputs)
    npt.assert_allclose(result, expected, rtol=1e-5)

def test_calculate_stokes_parameters_unpolarized():
    # P=0 → both Q and U should be exactly zero.
    t, q, u = calculate_stokes_parameters(10.0, 0.0, 30.0)
    assert t == 10.0
    assert q == 0
    assert u == 0

def test_calculate_stokes_parameters_45_degrees():
    # angle=45° → cos(2*45°)=0, sin(2*45°)=1 → Q≈0 (zeroed by threshold), U≠0.
    t, q, u = calculate_stokes_parameters(10.0, 50.0, 45.0)
    assert q == 0  # Below threshold, zeroed out
    assert u == pytest.approx(5.0)

def test_calculate_stokes_parameters_threshold_zeroing():
    # Custom large threshold forces both Q and U to zero.
    t, q, u = calculate_stokes_parameters(10.0, 19.0, 145.5, threshold=10.0)
    assert q == 0
    assert u == 0
