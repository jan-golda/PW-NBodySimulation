import sys
from typing import Dict, Type, Callable, Tuple

import numpy as np
from mpi4py import MPI

from benchmarking import benchmark
from distributions import center_of_mass_distribution
from simulations.brute_force import BruteForceSimulation
from simulations.core import Simulation
from simulations.distributed import DistributedSimulation
from simulations.parallel import ParallelSimulation
from simulations.sequential import SequentialSimulation
from visualization import Visualization
from arguments import PARSER

# available simulation methods
SIMULATIONS: Dict[str, Type[Simulation]] = {
    'brute': BruteForceSimulation,
    'sequential': SequentialSimulation,
    'parallel': ParallelSimulation,
    'distributed': DistributedSimulation
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

    # for MPI stop them here and put to work!
    if isinstance(simulation, DistributedSimulation) and MPI.COMM_WORLD.Get_rank() != 0:
        sim = simulation.simulate()
        while True:
            next(sim)

    # benchmarking
    if params.benchmark:
        benchmark(simulation, params)
        sys.exit()

    # visualization
    visualization = Visualization(
        simulation=simulation,
        max_fps=params.max_fps,
        title=f'Simulation - {params.method} - {params.bodies} bodies',
        show_trails=params.show_trails
    )

    visualization.run()
