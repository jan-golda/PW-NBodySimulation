import numpy as np

from typing import NamedTuple


class DistributedData(NamedTuple):
    """ Stores all the data needed for distributed implementation of Barnes-Hut. """
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    masses: np.ndarray

    nodes_positions: np.ndarray
    nodes_masses: np.ndarray
    nodes_sizes: np.ndarray
    nodes_children_types: np.ndarray
    nodes_children_ids: np.ndarray
