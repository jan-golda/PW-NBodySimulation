import numpy as np

from simulations.parallel.shared_data import SharedData
from simulations.utils import gravitational_force

OCTANT_EMPTY = 0
OCTANT_BODY = 1
OCTANT_NODE = 2

shared_data: SharedData


def initialize(data: SharedData):
    """ Initializes the worker with given shared data. """
    global shared_data
    shared_data = data


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
