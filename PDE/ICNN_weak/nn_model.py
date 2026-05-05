# Re-export from sibling ICNN folder to avoid duplication
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ICNN'))
from nn_model import ICNN  # noqa: F401
