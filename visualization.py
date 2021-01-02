import time

from vedo import Plotter, Points, Cube, Sphere

from simulations.core import Simulation


class Visualization:
    """ Visualization of a 3D N-Body simulation. """

    def __init__(self, simulation: Simulation, max_fps: float, title: str = 'Simulation',
                 size_multiplier: float = 0.02, trail_length: float = 2.0,
                 show_trails: bool = True):
        self._simulation = simulation
        self._min_step_time = 1.0 / max_fps

        # display setup
        self._plotter = Plotter(interactive=False, title=title)

        # bodies setup
        sizes = simulation.masses * size_multiplier
        self._bodies = [Sphere(pos, r=r, c='black') for pos, r in zip(simulation.positions, sizes)]
        self._plotter += self._bodies

        # trails
        if show_trails:
            for body in self._bodies:
                body.addTrail(alpha=0.3, lw=1, maxlength=trail_length, n=100)

        self._plotter.show(resetcam=True)

    def run(self):
        """ Runs the visualization of the simulation. """
        last_timestamp = time.time()

        for positions, velocities, accelerations in self._simulation.simulate():

            # throttle the simulation
            delta_time = time.time() - last_timestamp
            if delta_time < self._min_step_time:
                time.sleep(self._min_step_time - delta_time)
            last_timestamp = time.time()

            # update bodies
            for body, pos in zip(self._bodies, positions):
                body.pos(pos)

            # update display
            self._plotter.show(resetcam=False)
