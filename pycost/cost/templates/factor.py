"""
Factor model implementation.

This module provides a model for applying cost factors:
- Factor: Model for applying cost factors to base estimates
"""

import logging
import pandas as pd
import param

from pycost.cost.core import Model

# Get logger
logger = logging.getLogger(__name__)


class Factor(Model):
    """
    Model for applying cost factors to base estimates.
    
    This class provides a framework for applying various cost factors
    such as engineering factors, management factors, and inflation
    to base cost estimates.
    
    Attributes:
        base_model (Model): The base model to apply factors to
        factors (Dict[str, float]): Dictionary of factor names and values
        description (str): Description of the factors being applied
    """
    
    base_model = param.ClassSelector(Model, default=None)
    factors = param.Dict(default={})
    description = param.String(default="Cost Factors")
    
    def calc_cost(self) -> pd.DataFrame:
        """
        Calculate costs by applying factors to the base model.
        
        Returns:
            DataFrame with factored cost estimate
        """
        logger.debug("Calculating factored costs")
        
        if self.base_model is None:
            logger.warning("No base model specified for Factor model")
            self.cost_estimate = pd.DataFrame(columns=self.global_inputs.required_fields)
            return self.cost_estimate
            
        # Get the base cost estimate
        self.base_model.calc_cost()
        base_costs = self.base_model.cost_estimate.copy()
        
        # Apply factors to the base costs
        for factor_name, factor_value in self.factors.items():
            base_costs[factor_name] = factor_value
            base_costs['value_cp'] = base_costs['value_cp'] * factor_value
            
        self.cost_estimate = base_costs
        return self.cost_estimate 