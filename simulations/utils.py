from typing import Tuple

import numpy as np

GRAVITATIONAL_CONSTANT: float = 6.67408e-11


def gravitational_force(pos1: np.ndarray, pos2: np.ndarray,
                        mass1: float, mass2: float,
                        g: float = GRAVITATIONAL_CONSTANT, softening: float = 0.0
                        ) -> np.ndarray:
    """
    Calculates force applied on body 1 (pos1, mass1) exerted by body 2 (pos2, mass2).
    Args:
        pos1: Position of body 1 in form of a float vector of size (3,).
        pos2: Position of body 2 in form of a float vector of size (3,).
        mass1: Mass of body 1.
        mass2: Mass of body 2.
        g: Newton's gravitational constant.
        softening: Small value added to the calculated distance between bodies that prevents the force from
            going into infinity when the bodies are very close (caused by the numerical instability).

    Returns:
        Vector of force acting on body 1 in form of a float vector of size (3,).
    """
    vector = pos2 - pos1
    distance = np.linalg.norm(vector) + softening
    return g * mass1 * mass2 * vector / (distance ** 3)


def octant_coords(coords_min: np.ndarray, coords_max: np.ndarray, octant: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates coordinates of the specified octant in given space.
    Args:
        coords_min: Minimal coordinates of the region (float (3,)).
        coords_max: Maximal coordinates of the region (float (3,)).
        octant: Number of octant [0..8].

    Returns:
        (octant_min, octant_max) - octant coordinates.
    """
    assert octant in range(8)

    # decode octant into binary
    indices = (octant / np.array([1, 2, 4])).astype(np.int) % 2

    coords = np.array([
        coords_min,
        (coords_min + coords_max) / 2,
        coords_max
    ])

    return coords[indices, [0, 1, 2]], coords[indices+1, [0, 1, 2]]
