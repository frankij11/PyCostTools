#import pkg_resources
#my_data = pkg_resources.resource_string(__name__, "data/inflation.csv")
#print(my_data)


from .utils import *
from .inflation import *
from .learn import *
from .analysis import *

import pycost.cost_model

