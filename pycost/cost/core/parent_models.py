"""
Parent model implementations for cost estimation.

This module provides container classes for combining multiple cost models:
- ParentModel: Container for multiple cost models
"""

import logging
from typing import Any, List, Optional, Union
import pandas as pd
import param

from pycost.cost.core.base import Model

# Get logger
logger = logging.getLogger(__name__)


class ParentModel(Model):
    """
    Container class for combining multiple cost estimation models.
    
    This class allows multiple cost models to be combined into a single
    hierarchical model, with automatic aggregation of results.
    
    Attributes:
        models (List[Model]): List of child cost models
    """
    
    models = param.List()
    
    def __init__(self, **params):
        """
        Initialize the parent model and set parent paths for child models.
        """
        super(ParentModel, self).__init__(**params)
        
        # Set parent paths for all initial models
        if self.models:
            current_path = self.__class__.__name__
            if hasattr(self, "_parent_path") and self._parent_path:
                current_path = f"{self._parent_path} > {current_path}"
                
            for model in self.models:
                model.parent_path = current_path

    def calc_cost(self) -> pd.DataFrame:
        """
        Calculate costs for all child models.
        
        Returns:
            DataFrame containing the combined cost estimate
        """
        results = []
        logger.info("Running sequential cost calculations")

        current_path = self.__class__.__name__
        if self.parent_path:
            current_path = f"{self.parent_path} > {current_path}"
        else:
            current_path = current_path
            
        for model in self.models:
            # Set the parent path for each child model
            model.parent_path = current_path
            
            model.calc_cost()
            cost = model._cost_estimate
            results.append(cost)

        if len(results) > 0:
            self.cost_estimate = pd.concat(results, ignore_index=True)
            logger.debug(f"Combined {len(results)} model results")
        else:
            self.cost_estimate = pd.DataFrame(columns=self.global_inputs.required_fields)
    
        return self.cost_estimate
    
    def calc_cost_model(self, i: int = None, model: Model = None) -> None:
        """
        Calculate cost for a specific child model.
        
        Args:
            i: Index of the model to calculate
            model: Model object to calculate costs for
            
        Raises:
            ValueError: If neither i nor model is provided in a ParentModel with no models
            IndexError: If the provided index is out of range
        """
        if i is not None:
            try:
                self.models[i].calc_cost()
            except IndexError:
                logger.error(f"Model index {i} out of range (0-{len(self.models)-1})")
                raise IndexError(f"Model index {i} out of range (0-{len(self.models)-1})")
        elif model is not None:
            model.calc_cost()
        else:
            if not self.models:
                logger.warning("No models to calculate costs for")
                return
            
            # Calculate costs for all models
            for model in self.models:
                model.calc_cost()
    
    def get_model(self, i: int = None, model_name: str = None) -> Model:
        """
        Get a specific child model.
        
        Args:
            i: Index of the model to get
            model_name: Name of the model to get
            
        Returns:
            The requested model
            
        Raises:
            ValueError: If neither i nor model_name is provided, or if the model is not found
        """
        if i is not None:
            return self.models[i]
        elif model_name:
            for model in self.models:
                if model.name == model_name:
                    return model
            raise ValueError(f"Model {model_name} not found")
        else:
            raise ValueError("No model specified")

    def add_model(self, model: Model) -> None:
        """
        Add a new model to the parent model.
        
        Args:
            model: The model to add
        """
        logger.info(f"Adding model: {type(model).__name__}")
        
        # Set the parent path for the new model
        current_path = self.__class__.__name__
        if self.parent_path:
            current_path = f"{self.parent_path} > {current_path}"
        model.parent_path = current_path
        
        self.models.append(model)
        self.calc_cost()
    
    def _prepare_sim(self) -> None:
        """Prepare all child models for simulation."""
        if self.uncertainty_inputs is not None:
            logger.debug("Updating parameters for simulation")
            self.param.update(**self.uncertainty_inputs)
        for model in self.models:
            model._prepare_sim()
    
    def _end_sim(self) -> None:
        """Clean up after simulation for all child models."""
        if self.uncertainty_inputs is not None:
            logger.debug("Resetting parameters after simulation")
            for key, val in self.uncertainty_inputs.items():
                self.param.update(**{key: self.param[key].default})
        
        for model in self.models:
            model._end_sim()
        self.calc_cost() 