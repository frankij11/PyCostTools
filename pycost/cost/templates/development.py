"""
Development phase cost model implementation.

This module provides a model for development phase costs:
- Development: Model for development phase costs with annual phasing
"""

import logging
import pandas as pd
import param

from pycost.cost.core import Model

# Get logger
logger = logging.getLogger(__name__)


class Development(Model):
    """
    Model for development phase costs.
    
    Attributes:
        cost (float): Total development cost
        duration (int): Development duration in years
        start_year (int): Development start year
        phased_estimate (pd.DataFrame): Phased cost estimate
    """
    
    cost = param.Number(100)
    duration = param.Number(5)
    start_year = param.Number(2020)
    phased_estimate = pd.DataFrame()

    def __call__(self) -> pd.DataFrame:
        """Return the phased cost estimate."""
        return self.phased_estimate
    
    @property
    def end_year(self) -> int:
        """Calculate the development end year."""
        return self.start_year + self.duration

    @param.depends('start_year', 'duration', 'cost', watch=True, on_init=True)
    def calc_cost(self) -> pd.DataFrame:
        """
        Calculate the development cost estimate.
        
        Returns:
            DataFrame with annual development costs
        """
        logger.debug("Calculating development costs")
        df = pd.DataFrame(dict(
            FY=range(int(self.start_year), int(self.end_year)),
            value_cp=[self.cost / self.duration] * (int(self.end_year) - int(self.start_year))
        ))
        self.cost_estimate = df
        return self.cost_estimate 