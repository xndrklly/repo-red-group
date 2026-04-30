"""
sindy_torch — PyTorch implementation of SINDy
(Sparse Identification of Nonlinear Dynamics)

Key components:
  Library builders:  PolynomialLibrary, FourierLibrary, CompositeLibrary, CustomLibrary
  Solvers:           stls (classical), SparseOptimizer, GradientOptimizer
  Models:            SINDyModule, NeuralODEModule, ODEModel
  Differentiation:   finite_difference_4th, autograd_derivative, tv_reg_diff
  Reference systems: lorenz, hopf, logistic
"""

# Library builders
from .library import (
    LibraryBase,
    PolynomialLibrary,
    FourierLibrary,
    CompositeLibrary,
    CustomLibrary,
)

# Solvers
from .solvers import stls, stls_masked, SparseOptimizer, GradientOptimizer

# Models
from .models import SINDyModule, NeuralODEModule, ODEModel

# Differentiation
from .differentiation import finite_difference_4th, autograd_derivative, tv_reg_diff

# Reference dynamical systems
from .systems import lorenz, hopf, logistic

# Trajectory helpers
from .trajectory import (
    TrajectoryWindow,
    compute_state_scales,
    make_overlapping_trajectory_windows,
    normalized_state_mse,
    stack_state_trajectory,
)

# Utilities
from .utils import add_device_arg, as_numpy, get_device, to_tensor
