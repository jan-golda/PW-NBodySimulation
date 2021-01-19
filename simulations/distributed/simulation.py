from argparse import Namespace
from typing import Iterable, Tuple, List

import numpy as np
from mpi4py import MPI

from simulations.core import Simulation
from simulations.distributed.data import DistributedData
from simulations.utils import gravitational_force, octant_coords

OCTANT_EMPTY = 0
OCTANT_BODY = 1
OCTANT_NODE = 2


class DistributedSimulation(Simulation):
    """ Distributed (MPI) simulation of Barnes-Hut algorithm. """

    def __init__(self, positions: np.ndarray, velocities: np.ndarray,
                 masses: np.ndarray, params: Namespace):
        super().__init__(positions, velocities, masses, params)
        self._theta = params.theta

        self._init_memory(positions, velocities, masses)

        self._comm = MPI.COMM_WORLD
        self._comm_rank = self._comm.Get_rank()
        self._comm_size = self._comm.Get_size()

        # range of id this instance should operate on
        size, extras = divmod(self.bodies, self._comm_size)
        i = self._comm_rank
        begin = (size + 1) * min(i, extras) + size * max(0, i - extras)
        end = (size + 1) * min(i + 1, extras) + size * max(0, i + 1 - extras)
        self._ids_slice = slice(begin, end)

    def _init_memory(self, positions: np.ndarray, velocities: np.ndarray,
                     masses: np.ndarray):
        """ Prepares data arrays. """
        max_nodes = self.bodies + 64

        self._data = DistributedData(
            positions=np.array(positions, dtype=np.float),
            velocities=np.array(velocities, dtype=np.float),
            masses=np.array(masses, dtype=np.float),
            accelerations=np.zeros_like(velocities),
            nodes_positions=np.empty((max_nodes, 3), dtype=np.float),
            nodes_masses=np.empty((max_nodes,), dtype=np.float),
            nodes_sizes=np.empty((max_nodes,), dtype=np.float),
            nodes_children_types=np.empty((max_nodes, 8), dtype=np.int),
            nodes_children_ids=np.empty((max_nodes, 8), dtype=np.int)
        )

    def simulate(self) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """ Runs parallel implementation of Barnes-Hut simulation. """
        print(f'[{self._comm_rank}] SIMULATE')
        while True:
            self._build_octree()
            self._update_accelerations()
            self._update_positions()

            yield self._data.positions, self._data.velocities, self._data.accelerations

    def _build_octree(self):
        """ Builds octree used in Barnes-Hut. """
        if self._comm_rank == 0:
            # cleanup old tree
            self._nodes_positions = []
            self._nodes_masses = []
            self._nodes_sizes = []
            self._nodes_children_types = []
            self._nodes_children_ids = []

            min_pos = np.min(self._data.positions)
            max_pos = np.max(self._data.positions)

            self._build_octree_branch(
                bodies=list(range(self.bodies)),
                coords_min=np.array([min_pos] * 3),
                coords_max=np.array([max_pos] * 3)
            )

            self._data.nodes_positions[:len(self._nodes_positions)] = self._nodes_positions
            self._data.nodes_masses[:len(self._nodes_masses)] = self._nodes_masses
            self._data.nodes_sizes[:len(self._nodes_sizes)] = self._nodes_sizes
            self._data.nodes_children_types[:len(self._nodes_children_types)] = self._nodes_children_types
            self._data.nodes_children_ids[:len(self._nodes_children_ids)] = self._nodes_children_ids

        # synchronize initial data
        self._data.broadcast()

        self._comm.Barrier()

    def _build_octree_branch(self, bodies: List[int], coords_min: np.ndarray, coords_max: np.ndarray) -> Tuple[int, int]:
        """
        Builds a single branch of the octree.
        Args:
            bodies: Indices of the bodies that lay in this region.
            coords_min: Minimal coordinates of the region (float (3,)).
            coords_max: Maximal coordinates of the region (float (3,)).
        Returns:
            (octant_type, id):
                octant_type: one of OCTANT_EMPTY, OCTANT_BODY or OCTANT_NODE
                id: for OCTANT_EMPTY its -1, for OCTANT_BODY its body id and for OCTANT_NODE is node id
        """
        # in case of empty octant
        if len(bodies) == 0:
            return OCTANT_EMPTY, -1

        # in case of single body
        if len(bodies) == 1:
            return OCTANT_BODY, bodies[0]

        # create new node
        node_id = len(self._nodes_positions)
        self._nodes_positions.append(np.average(self._data.positions[bodies], axis=0, weights=self._data.masses[bodies]))
        self._nodes_masses.append(np.sum(self._data.masses[bodies]))
        self._nodes_sizes.append(coords_max[0] - coords_min[0])
        self._nodes_children_types.append(np.empty((8,), dtype=np.int))
        self._nodes_children_ids.append(np.empty((8,), dtype=np.int))

        # calculate octant for each body
        coords_mid = (coords_min + coords_max) / 2
        bodies_octant = np.sum((self._data.positions[bodies] > coords_mid) * [1, 2, 4], axis=1)

        # create octants
        for i in range(8):
            child_type, child_id = self._build_octree_branch(
                bodies=[body_id for body_id, octant in zip(bodies, bodies_octant) if octant == i],
                coords_min=octant_coords(coords_min, coords_max, i)[0],
                coords_max=octant_coords(coords_min, coords_max, i)[1]
            )
            self._nodes_children_types[node_id][i] = child_type
            self._nodes_children_ids[node_id][i] = child_id

        return OCTANT_NODE, node_id

    def _update_accelerations(self):
        """ Calculates accelerations of the bodies. """
        # reset accelerations
        self._data.accelerations[self._ids_slice] = 0.0

        # calculate accelerations
        for body_id in range(self._ids_slice.start, self._ids_slice.stop):
            self._update_body_acceleration(body_id, OCTANT_NODE, 0)

        # share results
        self._data.accelerations[:] = np.vstack(self._comm.allgather(self._data.accelerations[self._ids_slice]))

    def _update_body_acceleration(self, body_id: int, node_type: int, node_id: int):
        """ Updates acceleration of the given body using the given branch. """
        # in case of empty octant
        if node_type == OCTANT_EMPTY:
            return

        # in case of single body octant
        if node_type == OCTANT_BODY:
            # ignore if this is the same body
            if node_id == body_id:
                return

            position = self._data.positions[node_id]
            mass = self._data.masses[node_id]

        # in case of node octant
        else:
            # check the distance
            distance = np.linalg.norm(self._data.positions[body_id] - self._data.nodes_positions[node_id])
            node_size = self._data.nodes_sizes[node_id]

            # visit children if is not far enough
            if node_size / distance > self._theta:
                for child_type, child_id in zip(self._data.nodes_children_types[node_id], self._data.nodes_children_ids[node_id]):
                    self._update_body_acceleration(body_id, child_type, child_id)
                return

            # in other case treat as a single body
            position = self._data.nodes_positions[node_id]
            mass = self._data.nodes_masses[node_id]

        # calculate acceleration
        force = gravitational_force(
            pos1=self._data.positions[body_id],
            pos2=position,
            mass1=self._data.masses[body_id],
            mass2=mass,
            g=self.gravitational_constant,
            softening=self.softening
        )
        self._data.accelerations[body_id] += force / self._data.masses[body_id]

    def _update_positions(self):
        """ Calculates positions of the bodies. """
        # get data for this instance
        velocities = self._data.velocities[self._ids_slice]
        positions = self._data.positions[self._ids_slice]
        accelerations = self._data.accelerations[self._ids_slice]

        # calculate
        velocities += accelerations * self.time_step
        positions += velocities * self.time_step

        # share results
        self._data.velocities[:] = np.vstack(self._comm.allgather(velocities))
        self._data.positions[:] = np.vstack(self._comm.allgather(positions))

    @property
    def positions(self) -> np.ndarray:
        return self._data.positions

    @property
    def velocities(self) -> np.ndarray:
        return self._data.velocities

    @property
    def masses(self) -> np.ndarray:
        return self._data.masses

    @property
    def accelerations(self) -> np.ndarray:
        return self._data.accelerations
