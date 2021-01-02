from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Iterable, Tuple

import numpy as np


class Simulation(ABC):
    """ Interface for N-Body simulation. """

    def __init__(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, params: Namespace):
        """
        Initializes the simulation with given bodies.
        Args:
            positions: Float array of size (n,3) that describes initial positions of bodies.
            velocities: Float array of size (n,3) that describes initial velocities of bodies.
            masses: Float array of size (n,) that describes masses of bodies.
            params: Simulation parameters.
        """
        self._params = params

        assert positions.shape == (params.bodies, 3)
        assert velocities.shape == (params.bodies, 3)
        assert masses.shape == (params.bodies, )

    # ===================================================================
    #  Abstract methods
    # ===================================================================
    @abstractmethod
    def simulate(self) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Runs the simulation and yields bodies data after each simulation step.
        Yields: (positions, velocities, accelerations)
            positions: Float array of size (n,3) that describes current positions of bodies.
            velocities: Float array of size (n,3) that describes current velocities of bodies.
            accelerations: Float array of size (n,3) that describes current accelerations of bodies.
        """

    @property
    @abstractmethod
    def positions(self) -> np.ndarray:
        """ Current positions of the bodies. """

    @property
    @abstractmethod
    def velocities(self) -> np.ndarray:
        """ Current velocities of the bodies. """

    @property
    @abstractmethod
    def masses(self) -> np.ndarray:
        """ Masses of the bodies. """

    @property
    @abstractmethod
    def accelerations(self) -> np.ndarray:
        """ Current accelerations of the bodies. """

    # ===================================================================
    #  Settings getters
    # ===================================================================
    @property
    def bodies(self) -> int:
        """ Number of bodies in this simulation. """
        return self._params.bodies

    @property
    def time_step(self) -> float:
        """ Time step (dt) of the simulation. """
        return self._params.time_step

    @property
    def gravitational_constant(self) -> float:
        """ Newton's Gravitational Constant used in this simulation. """
        return self._params.gravitational_constant

    @property
    def softening(self) -> float:
        """ Softening parameter that prevents velocities from going into infinity. """
        return self._params.softening
