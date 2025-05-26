"""
Learning curve cost model implementation.

This module provides a model for learning curve analysis:
- LearningCurve: Model for manufacturing learning curve analysis
"""

import logging
import pandas as pd
import numpy as np
import param

from pycost.cost.core import Model

# Get logger
logger = logging.getLogger(__name__)


class LearningCurve(Model):
    """
    Model for manufacturing cost estimation using learning curves.
    
    Attributes:
        first_unit_cost (float): First unit cost
        learning_curve (float): Learning curve slope (0.7-1.0)
        rate_curve (float): Rate curve slope (0.7-1.0)
        quantity_profile (pd.DataFrame): Production quantity profile
    """
    
    first_unit_cost = param.Number(100)
    learning_curve = param.Number(.95, bounds=(.7, 1))
    rate_curve = param.Number(.95, bounds=(.7, 1))
    quantity_profile = param.DataFrame(pd.DataFrame(dict(
        FY=[2020]*10 + [2021]*20 + [2022]*15 + [2023]*5,
        Qty=np.arange(50) + 1,
        Rate=[10]*10 + [20]*20 + [15]*15 + [5]*5
    )))

    @param.depends('first_unit_cost', 'learning_curve', 'rate_curve', 'quantity_profile', watch=True)
    def calc_cost(self) -> None:
        """
        Calculate costs using learning curve analysis.
        
        This method applies learning and rate curves to the
        quantity profile to estimate manufacturing costs.
        """
        logger.debug("Calculating learning curve costs")
        n = self.quantity_profile.shape[0]
        tmp = self.quantity_profile.assign(
            Element=['Learn'] * n,
            APPN=['APN'] * n,
            BaseYear=[2020] * n
        ).assign(
            Value=lambda x: self.first_unit_cost * (x.Qty**(np.log(self.learning_curve))) * (x.Rate**(np.log(self.rate_curve))),
            value_cp=lambda x: x.Value
        )
        self.cost_estimate = tmp 