import argparse


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


PARSER = argparse.ArgumentParser(
    description=(
        'N-Body simulation with visualization'
    ),
    formatter_class=lambda prog: CustomFormatter(prog, max_help_position=100)
)

PARSER.add_argument(
    '-b', '--bodies',
    type=int,
    required=True,
    metavar='N',
    help='Number of bodies'
)

PARSER.add_argument(
    '-m', '--method',
    type=str,
    required=True,
    choices=['brute', 'sequential', 'parallel'],
    help='Simulation method'
)

PARSER.add_argument(
    '-d', '--distribution',
    type=str,
    required=True,
    choices=['center-of-mass'],
    help='Initial distribution of the bodies'
)

PARSER.add_argument(
    '-dt', '--time_step',
    type=float,
    default=0.01,
    metavar='DT',
    help='Time step of the simulation'
)

PARSER.add_argument(
    '-f', '--max_fps',
    type=float,
    default=30.0,
    metavar='FPS',
    help='Maximum number of frames per second'
)

PARSER.add_argument(
    '-g', '--gravitational_constant',
    type=float,
    default=1.0,
    metavar='G',
    help='Newton\'s Gravitational Constant'
)

PARSER.add_argument(
    '-s', '--softening',
    type=float,
    default=0.1,
    metavar='S',
    help='Softening parameter that prevents velocities from going into infinity'
)

PARSER.add_argument(
    '-t', '--theta',
    type=float,
    default=0.5,
    metavar='T',
    help='Theta parameter of the Barnes-Hut simulation'
)

PARSER.add_argument(
    '-p', '--processes',
    type=int,
    metavar='N',
    help='Number of processes used in parallel implementation. Defaults to number of available CPUs'
)

PARSER.add_argument(
    '--seed',
    type=int,
    metavar='SEED',
    help='Seed used for defining initial conditions.'
)

PARSER.add_argument(
    '--show_trails',
    action='store_true',
    help='Show body trails.'
)
