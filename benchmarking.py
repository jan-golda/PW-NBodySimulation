import time
from argparse import Namespace

import numpy as np
from progress.bar import Bar

from simulations.core import Simulation


def benchmark(simulation: Simulation, params: Namespace):
    """ Runs benchmark of given simulation. """

    it = simulation.simulate()
    times = []

    with Bar('Benchmarking', max=params.benchmark_steps) as bar:
        for i in range(params.benchmark_steps):
            start_time = time.time()
            next(it)
            times.append(time.time() - start_time)

            bar.next()

    times = np.array(times)

    print(f'Mean step time:  {times.mean():.4f}')
    print(f'Std step time:   {times.std():.4f}')
    print(f'Max step time:   {times.max():.4f}')
    print(f'Min step time:   {times.min():.4f}')
