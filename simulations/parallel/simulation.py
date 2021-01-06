import atexit
from argparse import Namespace
from multiprocessing import Pool, Value
from multiprocessing.managers import SharedMemoryManager
from typing import Iterable, Tuple

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

        max_nodes = self.bodies + 64

        # create shared memory buffers
        self._positions_shm = self._memory_manager.SharedMemory(positions.nbytes)
        self._velocities_shm = self._memory_manager.SharedMemory(velocities.nbytes)
        self._accelerations_shm = self._memory_manager.SharedMemory(velocities.nbytes)
        self._masses_shm = self._memory_manager.SharedMemory(masses.nbytes)

        self._nodes_positions_shm = self._memory_manager.SharedMemory(np.empty((max_nodes, 3), np.float).nbytes)
        self._nodes_masses_shm = self._memory_manager.SharedMemory(np.empty((max_nodes, ), np.float).nbytes)
        self._nodes_sizes_shm = self._memory_manager.SharedMemory(np.empty((max_nodes, ), np.float).nbytes)
        self._nodes_children_types_shm = self._memory_manager.SharedMemory(np.empty((max_nodes, 8), np.int).nbytes)
        self._nodes_children_ids_shm = self._memory_manager.SharedMemory(np.empty((max_nodes, 8), np.int).nbytes)

        # setup NumPy arrays
        self._data = SharedData(
            time_step=self.time_step,
            theta=self._theta,
            gravitational_constant=self.gravitational_constant,
            softening=self.softening,

            nodes_count=Value('i', 0),

            positions=np.ndarray((self.bodies, 3), dtype=np.float, buffer=self._positions_shm.buf),
            velocities=np.ndarray((self.bodies, 3), dtype=np.float, buffer=self._velocities_shm.buf),
            accelerations=np.ndarray((self.bodies, 3), dtype=np.float, buffer=self._accelerations_shm.buf),
            masses=np.ndarray((self.bodies, ), dtype=np.float, buffer=self._masses_shm.buf),

            nodes_positions=np.ndarray((max_nodes, 3), dtype=np.float, buffer=self._nodes_positions_shm.buf),
            nodes_masses=np.ndarray((max_nodes, ), dtype=np.float, buffer=self._nodes_masses_shm.buf),
            nodes_sizes=np.ndarray((max_nodes, ), dtype=np.float, buffer=self._nodes_sizes_shm.buf),
            nodes_children_types=np.ndarray((max_nodes, 8), dtype=np.int, buffer=self._nodes_children_types_shm.buf),
            nodes_children_ids=np.ndarray((max_nodes, 8), dtype=np.int, buffer=self._nodes_children_ids_shm.buf)
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
        global_coords_min = np.repeat(np.min(self._data.positions), 3)
        global_coords_max = np.repeat(np.max(self._data.positions), 3)
        global_coords_mid = (global_coords_min + global_coords_max) / 2

        # manually build first node
        self._data.nodes_count.value = 1
        self._data.nodes_positions[0] = np.average(self._data.positions, axis=0, weights=self._data.masses)
        self._data.nodes_masses[0] = np.sum(self._data.masses)
        self._data.nodes_sizes[0] = global_coords_max[0] - global_coords_min[0]

        # calculate base octant for each body
        bodies_base_octant = np.sum((self._data.positions > global_coords_mid) * [1, 2, 4], axis=1)

        tasks_targets = []
        tasks_args = []

        # build second layer of nodes and collect tasks
        for octant in range(8):
            coords_min, coords_max = octant_coords(global_coords_min, global_coords_max, octant)
            coords_mid = (coords_min + coords_max) / 2

            # get indices of bodies in this octant
            octant_bodies = np.argwhere(bodies_base_octant == octant).flatten()

            # if node is empty or has one body handle it separately
            if octant_bodies.size == 0:
                self._data.nodes_children_types[0, octant] = OCTANT_EMPTY
                continue
            if octant_bodies.size == 1:
                self._data.nodes_children_types[0, octant] = OCTANT_BODY
                self._data.nodes_children_ids[0, octant] = octant_bodies[0]
                continue

            # create node
            node_id = self._data.nodes_count.value
            self._data.nodes_count.value = node_id + 1
            self._data.nodes_children_types[0, octant] = OCTANT_NODE
            self._data.nodes_children_ids[0, octant] = node_id

            self._data.nodes_positions[node_id] = np.average(self._data.positions[octant_bodies], axis=0, weights=self._data.masses[octant_bodies])
            self._data.nodes_masses[node_id] = np.sum(self._data.masses[octant_bodies])
            self._data.nodes_sizes[node_id] = coords_max[0] - coords_min[0]

            # split bodies into sub octants
            bodies_sub_octant = np.sum((self._data.positions[octant_bodies] > coords_mid) * [1, 2, 4], axis=1)

            # create tasks
            for i in range(8):
                tasks_targets.append((node_id, i))
                tasks_args.append((
                    octant_bodies[bodies_sub_octant == i],
                    *octant_coords(coords_min, coords_max, i)
                ))

        # run tasks
        results = self._pool.starmap(worker.build_octree_branch, tasks_args)

        # update references in nodes
        for (node_id, i), (sub_node_type, sub_node_id) in zip(tasks_targets, results):
            self._data.nodes_children_types[node_id, i] = sub_node_type
            self._data.nodes_children_ids[node_id, i] = sub_node_id

    def _update_accelerations(self):
        """ Calculates accelerations of the bodies. """
        if self.bodies < 2:
            return

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
