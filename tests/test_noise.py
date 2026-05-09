import numpy as np
import pytest
from pulsar import MultiplicativeGaussianNoise
from pulsar.noise import Noise


def test_noise_is_abstract():
    # Direct instantiation of the abstract base must raise.
    with pytest.raises(TypeError):
        Noise()


class TestMultiplicateGaussianNoise:

    def test_to_dict(self):
        noise = MultiplicativeGaussianNoise(mean=2.0, std=0.2)
        assert noise.as_dict() == {'mean': 2.0, 'std': 0.2}

    def test_apply(self):
        np.random.seed(42)
        noise = MultiplicativeGaussianNoise(mean=1.0, std=0.2)
        data = np.array([[1, 2, 3]], dtype=float)
        noise.apply(data)
        assert data.shape == (1,3)
        assert np.all(data != [[1, 2, 3]])
        assert np.all(data > 0.7)
        assert np.all(data < 3.4)