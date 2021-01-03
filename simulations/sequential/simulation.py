from argparse import Namespace
from typing import Iterable, Tuple, List

import numpy as np

from simulations.core import Simulation

# octant types
from simulations.utils import octant_coords, gravitational_force

OCTANT_EMPTY = 0
OCTANT_BODY = 1
OCTANT_NODE = 2


class SequentialSimulation(Simulation):
    """ Sequential simulation of Barnes-Hut algorithm. """

    def __init__(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, params: Namespace):
        super().__init__(positions, velocities, masses, params)
        self._theta = params.theta

        # initialize data
        self._positions = np.array(positions, dtype=np.float)
        self._velocities = np.array(velocities, dtype=np.float)
        self._masses = np.array(masses, dtype=np.float)
        self._accelerations = np.zeros_like(self._velocities)

        self._nodes_position = []
        self._nodes_masse = []
        self._nodes_size = []
        self._nodes_children_type = []
        self._nodes_children_id = []

    def simulate(self) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """ Runs sequential implementation of Barnes-Hut simulation. """
        while True:
            self._build_octree()
            self._update_accelerations()
            self._update_positions()

            yield self._positions, self._velocities, self._accelerations

    def _build_octree(self):
        """ Builds octree used in Barnes-Hut. """

        # cleanup old tree
        self._nodes_position = []
        self._nodes_mass = []
        self._nodes_size = []
        self._nodes_children_type = []
        self._nodes_children_id = []

        min_pos = np.min(self._positions)
        max_pos = np.max(self._positions)

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
        node_id = len(self._nodes_position)
        self._nodes_position.append(np.average(self._positions[bodies], axis=0, weights=self._masses[bodies]))
        self._nodes_mass.append(np.sum(self._masses[bodies]))
        self._nodes_size.append(coords_max[0] - coords_min[0])
        self._nodes_children_type.append(np.empty((8,), dtype=np.int))
        self._nodes_children_id.append(np.empty((8,), dtype=np.int))

        # calculate octant for each body
        coords_mid = (coords_min + coords_max) / 2
        bodies_octant = np.sum((self._positions[bodies] > coords_mid) * [1, 2, 4], axis=1)

        # create octants
        for i in range(8):
            child_type, child_id = self._build_octree_branch(
                bodies=[body_id for body_id, octant in zip(bodies, bodies_octant) if octant == i],
                coords_min=octant_coords(coords_min, coords_max, i)[0],
                coords_max=octant_coords(coords_min, coords_max, i)[1]
            )
            self._nodes_children_type[node_id][i] = child_type
            self._nodes_children_id[node_id][i] = child_id

        return OCTANT_NODE, node_id

    def _update_accelerations(self):
        """ Calculates accelerations of the bodies. """
        self._accelerations[:] = 0

        if self.bodies < 2:
            return

        for body_id in range(self.bodies):
            self._update_body_acceleration(body_id, OCTANT_NODE, 0)

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

            position = self._positions[node_id]
            mass = self._masses[node_id]

        # in case of node octant
        else:
            # check the distance
            distance = np.linalg.norm(self._positions[body_id] - self._nodes_position[node_id])
            node_size = self._nodes_size[node_id]

            # visit children if is not far enough
            if node_size / distance > self._theta:
                for child_type, child_id in zip(self._nodes_children_type[node_id], self._nodes_children_id[node_id]):
                    self._update_body_acceleration(body_id, child_type, child_id)
                return

            # in other case treat as a single body
            position = self._nodes_position[node_id]
            mass = self._nodes_mass[node_id]

        # calculate acceleration
        force = gravitational_force(
            pos1=self._positions[body_id],
            pos2=position,
            mass1=self._masses[body_id],
            mass2=mass,
            g=self.gravitational_constant,
            softening=self.softening
        )
        self._accelerations[body_id] += force / self._masses[body_id]

    def _update_positions(self):
        """ Calculates positions of the bodies. """
        self._velocities += self._accelerations * self.time_step
        self._positions += self._velocities * self.time_step

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
