"""
PyCost - Python-based cost estimation and analysis tools

This package provides utilities and models for cost estimation, learning curves, 
inflation adjustments, and other cost analysis tools commonly used in program management.
"""

__version__ = "0.1.0"
__author__ = "PyCost Team"

#import pkg_resources
#my_data = pkg_resources.resource_string(__name__, "data/inflation.csv")
#print(my_data)
"""
# Basic utilities
try:
    import pycost.utils
except ImportError:
    print("Warning: Could not import pycost.utils")

# Analysis modules
try:
    import pycost.inflation
except ImportError:
    print("Warning: Could not import pycost.inflation")

try:
    import pycost.learn
except ImportError:
    print("Warning: Could not import pycost.learn")

try:
    import pycost.analysis
except ImportError:
    print("Warning: Could not import pycost.analysis")

try:
    import pycost.analysis.process
except ImportError:
    print("Warning: Could not import pycost.process")
"""
# Import core modules
from pycost import utils

# Import analysis modules
from pycost import analysis

# Import cost modules
from pycost import cost

# Import inflation modules
from pycost import inflation