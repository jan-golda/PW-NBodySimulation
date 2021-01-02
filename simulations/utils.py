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
