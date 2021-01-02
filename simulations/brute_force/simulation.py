import itertools
from argparse import Namespace
from typing import Iterable, Tuple

import numpy as np

from simulations.core import Simulation
from simulations.core.utils import gravitational_force


class BruteForceSimulation(Simulation):
    """ Brute-Force (n^2) implementation of n-body simulation. """

    def __init__(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, params: Namespace):
        super().__init__(positions, velocities, masses, params)

        # initialize data
        self._positions = np.array(positions, dtype=np.float)
        self._velocities = np.array(velocities, dtype=np.float)
        self._masses = np.array(masses, dtype=np.float)
        self._accelerations = np.zeros_like(self._velocities)

    def simulate(self) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """ Runs Brute-Force simulation. """
        while True:

            # reset accelerations
            self._accelerations[:] = 0.0

            # calculate accelerations
            for b1, b2 in itertools.combinations(range(self.bodies), 2):
                force = gravitational_force(
                    pos1=self._positions[b1],
                    pos2=self._positions[b2],
                    mass1=self._masses[b1],
                    mass2=self._masses[b2],
                    g=self.gravitational_constant,
                    softening=self.softening
                )
                self._accelerations[b1] += force / self._masses[b1]
                self._accelerations[b2] -= force / self._masses[b2]

            # update velocities and positions
            self._velocities += self._accelerations * self.time_step
            self._positions += self._velocities * self.time_step

            # return simulation results
            yield self._positions, self._velocities, self._accelerations

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def velocities(self) -> np.ndarray:
        return self._velocities

    @property
    def masses(self) -> np.ndarray:
        return self._masses

    @property
    def accelerations(self) -> np.ndarray:
        return self._accelerations
