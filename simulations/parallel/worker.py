from typing import List, Tuple

import numpy as np

from simulations.parallel.shared_data import SharedData
from simulations.utils import gravitational_force, octant_coords

OCTANT_EMPTY = 0
OCTANT_BODY = 1
OCTANT_NODE = 2

shared_data: SharedData


def initialize(data: SharedData):
    """ Initializes the worker with given shared data. """
    global shared_data
    shared_data = data


def build_octree_branch(bodies: List[int], coords_min: np.ndarray, coords_max: np.ndarray) -> Tuple[int, int]:
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

    # get a id for a new node
    with shared_data.nodes_count.get_lock():
        node_id = shared_data.nodes_count.value
        shared_data.nodes_count.value = node_id + 1

    # create new node
    shared_data.nodes_positions[node_id] = np.average(shared_data.positions[bodies], axis=0, weights=shared_data.masses[bodies])
    shared_data.nodes_masses[node_id] = np.sum(shared_data.masses[bodies])
    shared_data.nodes_sizes[node_id] = coords_max[0] - coords_min[0]

    # calculate octant for each body
    coords_mid = (coords_min + coords_max) / 2
    bodies_octant = np.sum((shared_data.positions[bodies] > coords_mid) * [1, 2, 4], axis=1)

    # create octants
    for i in range(8):
        child_type, child_id = build_octree_branch(
            bodies=[body_id for body_id, octant in zip(bodies, bodies_octant) if octant == i],
            coords_min=octant_coords(coords_min, coords_max, i)[0],
            coords_max=octant_coords(coords_min, coords_max, i)[1]
        )
        shared_data.nodes_children_types[node_id, i] = child_type
        shared_data.nodes_children_ids[node_id, i] = child_id

    return OCTANT_NODE, node_id


def update_acceleration(body_id: int):
    """ Updates acceleration of the body. """
    shared_data.accelerations[body_id, :] = 0.0
    _update_body_acceleration(body_id, OCTANT_NODE, 0)


def _update_body_acceleration(body_id: int, node_type: int, node_id: int):
    """ Updates acceleration of the given body using the given branch. """
    # in case of empty octant
    if node_type == OCTANT_EMPTY:
        return

    # in case of single body octant
    if node_type == OCTANT_BODY:
        # ignore if this is the same body
        if node_id == body_id:
            return

        position = shared_data.positions[node_id]
        mass = shared_data.masses[node_id]

    # in case of node octant
    else:
        # check the distance
        distance = np.linalg.norm(shared_data.positions[body_id] - shared_data.nodes_positions[node_id])
        node_size = shared_data.nodes_sizes[node_id]

        # visit children if is not far enough
        if node_size / distance > shared_data.theta:
            for child_type, child_id in zip(shared_data.nodes_children_types[node_id], shared_data.nodes_children_ids[node_id]):
                _update_body_acceleration(body_id, child_type, child_id)
            return

        # in other case treat as a single body
        position = shared_data.nodes_positions[node_id]
        mass = shared_data.nodes_masses[node_id]

    # calculate acceleration
    force = gravitational_force(
        pos1=shared_data.positions[body_id],
        pos2=position,
        mass1=shared_data.masses[body_id],
        mass2=mass,
        g=shared_data.gravitational_constant,
        softening=shared_data.softening
    )
    shared_data.accelerations[body_id] += force / shared_data.masses[body_id]


def update_position(body_id: int):
    """ Updates position of a the body. """
    shared_data.velocities[body_id] += shared_data.accelerations[body_id] * shared_data.time_step
    shared_data.positions[body_id] += shared_data.velocities[body_id] * shared_data.time_step
