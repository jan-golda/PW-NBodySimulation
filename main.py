import argparse
from typing import Dict, Type, Callable, Tuple

import numpy as np

from distributions import center_of_mass_distribution
from simulations.brute_force import BruteForceSimulation
from simulations.core import Simulation
from visualization import Visualization
from arguments import PARSER

# available simulation methods
SIMULATIONS: Dict[str, Type[Simulation]] = {
    'brute': BruteForceSimulation
}

# available initial distributions
DISTRIBUTIONS: Dict[str, Callable[[int], Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
    'center-of-mass': center_of_mass_distribution
}

if __name__ == '__main__':
    params = PARSER.parse_args()

    # set seed
    if params.seed is not None:
        np.random.seed(params.seed)

    positions, velocities, masses = DISTRIBUTIONS[params.distribution](params.bodies)

    simulation = SIMULATIONS[params.method](
        positions=positions,
        velocities=velocities,
        masses=masses,
        params=params
    )

    visualization = Visualization(
        simulation=simulation,
        max_fps=params.max_fps,
        title=f'Simulation - {params.method} - {params.bodies} bodies',
        show_trails=params.show_trails
    )

    visualization.run()
