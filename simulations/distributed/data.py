import numpy as np

from typing import NamedTuple

from mpi4py import MPI


class DistributedData(NamedTuple):
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    masses: np.ndarray

    nodes_positions: np.ndarray
    nodes_masses: np.ndarray
    nodes_sizes: np.ndarray
    nodes_children_types: np.ndarray
    nodes_children_ids: np.ndarray

    def broadcast(self):
        for data in self:
            MPI.COMM_WORLD.Bcast(data, root=0)
