from .reference import lorenz, hopf, logistic
from .spring_lattice import (
    SpringLatticeODE,
    make_no_forcing,
    make_constant_node_force,
    make_sinusoidal_node_force,
)