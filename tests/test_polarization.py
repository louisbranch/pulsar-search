import numpy.testing as npt
import pytest
from pulsar.polarization import calculate_stokes_parameters

@pytest.mark.parametrize("name, inputs, expected", [
    ('Crab in X-Ray', (10.0, 19.0, 145.5), (10.0, 0.680899, -1.773803))
])
def test_calculate_stokes_parameters(name, inputs, expected):
    result = calculate_stokes_parameters(*inputs)
    npt.assert_allclose(result, expected, rtol=1e-5)
