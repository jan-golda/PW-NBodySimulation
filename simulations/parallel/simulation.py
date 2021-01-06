import atexit
from argparse import Namespace
from multiprocessing import Pool
from multiprocessing.managers import SharedMemoryManager
from typing import Iterable, Tuple, List

import numpy as np

from simulations.parallel import worker
from simulations.core import Simulation
from simulations.parallel.shared_data import SharedData
from simulations.parallel.worker import OCTANT_EMPTY, OCTANT_BODY, OCTANT_NODE
from simulations.utils import octant_coords


class ParallelSimulation(Simulation):
    """ Parallel simulation of Barnes-Hut algorithm realised using shared memory. """

    def __init__(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, params: Namespace):
        super().__init__(positions, velocities, masses, params)
        self._theta = params.theta

        self._init_memory(positions, velocities, masses)
        self._init_workers()

        atexit.register(self._cleanup)

    def _init_memory(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray):
        """ Prepares shared memory arrays. """
        # setup process that sets up shared memory
        self._memory_manager = SharedMemoryManager()
        self._memory_manager.start()

        # create shared memory buffers
        self._positions_shm = self._memory_manager.SharedMemory(positions.nbytes)
        self._velocities_shm = self._memory_manager.SharedMemory(velocities.nbytes)
        self._accelerations_shm = self._memory_manager.SharedMemory(velocities.nbytes)
        self._masses_shm = self._memory_manager.SharedMemory(masses.nbytes)
        self._nodes_positions_shm = self._memory_manager.SharedMemory(positions.nbytes)
        self._nodes_masses_shm = self._memory_manager.SharedMemory(masses.nbytes)
        self._nodes_sizes_shm = self._memory_manager.SharedMemory(masses.nbytes)
        self._nodes_children_types_shm = self._memory_manager.SharedMemory(np.empty((self.bodies, 8), np.int).nbytes)
        self._nodes_children_ids_shm = self._memory_manager.SharedMemory(np.empty((self.bodies, 8), np.int).nbytes)

        # setup NumPy arrays
        self._data = SharedData(
            time_step=self.time_step,
            theta=self._theta,
            gravitational_constant=self.gravitational_constant,
            softening=self.softening,

            positions=np.ndarray((self.bodies, 3), dtype=np.float, buffer=self._positions_shm.buf),
            velocities=np.ndarray((self.bodies, 3), dtype=np.float, buffer=self._velocities_shm.buf),
            accelerations=np.ndarray((self.bodies, 3), dtype=np.float, buffer=self._accelerations_shm.buf),
            masses=np.ndarray((self.bodies, ), dtype=np.float, buffer=self._masses_shm.buf),

            nodes_positions=np.ndarray((self.bodies, 3), dtype=np.float, buffer=self._nodes_positions_shm.buf),
            nodes_masses=np.ndarray((self.bodies, ), dtype=np.float, buffer=self._nodes_masses_shm.buf),
            nodes_sizes=np.ndarray((self.bodies, ), dtype=np.float, buffer=self._nodes_sizes_shm.buf),
            nodes_children_types=np.ndarray((self.bodies, 8), dtype=np.int, buffer=self._nodes_children_types_shm.buf),
            nodes_children_ids=np.ndarray((self.bodies, 8), dtype=np.int, buffer=self._nodes_children_ids_shm.buf)
        )

        # copy data into shared arrays
        self._data.positions[:] = positions[:]
        self._data.velocities[:] = velocities[:]
        self._data.masses[:] = masses[:]

    def _init_workers(self):
        """ Prepares pool of workers. """
        self._pool = Pool(
            processes=self._params.processes,
            initializer=worker.initialize,
            initargs=(self._data, )
        )

    def _cleanup(self):
        """ Cleans up shared memory and pool of workers. """
        self._pool.terminate()
        self._memory_manager.shutdown()
        print('Memory manager was shut down.')

    def simulate(self) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """ Runs parallel implementation of Barnes-Hut simulation. """
        while True:
            self._build_octree()
            self._update_accelerations()
            self._update_positions()

            yield self._data.positions, self._data.velocities, self._data.accelerations

    def _build_octree(self):
        """ Builds octree used in Barnes-Hut. """

        # cleanup old tree
        self._nodes_count = 0

        min_pos = np.min(self._data.positions)
        max_pos = np.max(self._data.positions)

        self._build_octree_branch(
            bodies=list(range(self.bodies)),
            coords_min=np.array([min_pos] * 3),
            coords_max=np.array([max_pos] * 3)
        )

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
        node_id = self._nodes_count
        self._nodes_count += 1
        self._data.nodes_positions[node_id] = np.average(self._data.positions[bodies], axis=0, weights=self._data.masses[bodies])
        self._data.nodes_masses[node_id] = np.sum(self._data.masses[bodies])
        self._data.nodes_sizes[node_id] = coords_max[0] - coords_min[0]

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
            self._data.nodes_children_types[node_id, i] = child_type
            self._data.nodes_children_ids[node_id, i] = child_id

        return OCTANT_NODE, node_id

    def _update_accelerations(self):
        """ Calculates accelerations of the bodies. """
        self._pool.map(worker.update_acceleration, range(self.bodies))

    def _update_positions(self):
        """ Calculates positions of the bodies. """
        self._pool.map(worker.update_position, range(self.bodies))

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
