""" Functions that generate initial positions of the bodies. """
from typing import Tuple

import numpy as np


def center_of_mass_distribution(bodies: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Generates random initial distribution of the bodies that meets the center-of-mass frame. """
    positions = np.random.randn(bodies, 3)
    velocities = np.random.randn(bodies, 3)
    masses = 20.0 * np.ones((bodies,)) / bodies

    velocities -= np.mean(masses[:, np.newaxis] * velocities, 0) / np.mean(masses)

    return positions, velocities, masses
