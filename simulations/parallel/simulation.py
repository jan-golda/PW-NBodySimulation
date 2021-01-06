from argparse import Namespace
from typing import Iterable, Tuple

import numpy as np

from simulations.core import Simulation


class ParallelSimulation(Simulation):

    def __init__(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, params: Namespace):
        super().__init__(positions, velocities, masses, params)

    def simulate(self) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        pass

    @property
    def positions(self) -> np.ndarray:
        pass

    @property
    def velocities(self) -> np.ndarray:
        pass

    @property
    def masses(self) -> np.ndarray:
        pass

    @property
    def accelerations(self) -> np.ndarray:
        pass
