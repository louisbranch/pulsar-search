from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import numpy as np

class Noise(ABC):
    @abstractmethod
    def apply(self, data): # pragma: no cover
        pass

    def as_dict(self) -> dict:
        return asdict(self)

@dataclass
class MultiplicativeGaussianNoise(Noise):
    mean: float = 1.0
    std: float = 0.1
    
    def apply(self, data):
        size = data.shape[0]
        noise = np.random.normal(self.mean, self.std, size)
        noise = noise[:, None]
        data *= noise