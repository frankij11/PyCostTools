"""
Production phase cost model implementation.

This module provides a model for production phase costs:
- Production: Model for production phase costs using learning curves
"""

import logging
from typing import Any
import pandas as pd
import numpy as np
import param

from pycost.cost.core.base import Model
from pycost.cost.core.inventory import Inventory

# Get logger
logger = logging.getLogger(__name__)


class Production(Model):
    """
    Model for production phase costs using learning curve analysis.
    
    Attributes:
        first_unit_cost (float): First unit cost
        learning_curve (float): Learning curve slope (0.6-1.0)
        rate_curve (float): Rate curve slope (0.6-1.0)
        prior_units (float): Prior production quantity
        inventory (Inventory): Production inventory profile
    """
    
    first_unit_cost = param.Number(default=100)
    learning_curve = param.Number(default=.95, bounds=(.6, 1))
    rate_curve = param.Number(default=.95, bounds=(.6, 1))
    prior_units = param.Number(default=0)
    inventory = param.ClassSelector(Inventory, default=Inventory())

    def __call__(self) -> pd.DataFrame:
        """Return the lot cost estimate."""
        return self.lot_cost

    @property
    def quantities(self) -> pd.DataFrame:
        """Get the production quantities."""
        return self.inventory.procurement

    @param.depends('first_unit_cost', 'learning_curve', 'rate_curve', 'prior_units', 'inventory.profile', watch=False)
    def calc_lot_cost(self) -> pd.DataFrame:
        """
        Calculate production costs using learning curve analysis.
        
        Returns:
            DataFrame with lot costs
        """
        logger.debug("Calculating lot costs")
        df = (self
              .inventory.profile.assign(
                  T1=self.first_unit_cost,
                  LC=self.learning_curve,
                  RC=self.rate_curve,
                  Priors=self.prior_units
              )
        )
        df = (df
              .assign(
                  First=lambda x: x.quantity.cumsum().shift(1).fillna(1) + self.prior_units,
                  Last=lambda x: x.quantity.cumsum() + self.prior_units,
                  midpoint=lambda x: ((x.First + x.Last) * (x.First*x.Last))**.5 / 4,
                  auc=lambda x: x.T1 * x.midpoint**(np.log(x.LC)/2) * x.quantity**(np.log(x.RC)/2),
                  value_cp=lambda x: x.auc * x.quantity
              )
        )
        self.cost_estimate = df
        return self.cost_estimate
    
    @property
    def lot_cost(self) -> pd.DataFrame:
        """Get the lot cost estimate."""
        return self.calc_lot_cost()

    @param.depends('calc_lot_cost')
    def plot_lot_cost(self) -> Any:
        """Create a plot of lot costs."""
        return self.lot_cost.hvplot(x='FY', y='auc')

    @param.depends('first_unit_cost', 'learning_curve', 'rate_curve', 'prior_units', 'inventory.profile', watch=True)
    def calc_cost(self) -> pd.DataFrame:
        """
        Calculate production costs.
        
        Returns:
            DataFrame with production costs
        """
        logger.debug("Calculating production costs")
        self.cost_estimate = self.calc_lot_cost()
        return self.cost_estimate 