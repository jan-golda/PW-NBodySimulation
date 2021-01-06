from multiprocessing import Value
from typing import NamedTuple

import numpy as np


class SharedData(NamedTuple):
    """ Stores all the data needed for parallel implementation of Barnes-Hut. """
    time_step: float
    theta: float
    gravitational_constant: float
    softening: float

    nodes_count: Value

    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    masses: np.ndarray

    nodes_positions: np.ndarray
    nodes_masses: np.ndarray
    nodes_sizes: np.ndarray
    nodes_children_types: np.ndarray
    nodes_children_ids: np.ndarray
