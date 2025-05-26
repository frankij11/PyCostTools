"""
Base classes for cost estimation framework.

This module provides the fundamental classes for cost estimation:
- GlobalInputs: Program-wide parameters and settings
- Model: Base class for all cost estimation models
"""

import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import param
try:
    import panel as pn
except ImportError:
    pass

# Removed the direct import to avoid circular reference
# We'll import sim_tool dynamically when needed
from pycost.cost.utils.sim_tool import SimEngine, RVLognormal
from pycost.cost.utils.reactive import build_param_dependency_graph, check_param_cycles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create sim_engine later when needed
sim_engine = SimEngine()


class GlobalInputs(param.Parameterized):
    """
    Global parameters and settings for cost estimation.
    
    This class defines program-wide parameters that affect all cost models,
    including program identification, base year, and currency units.
    
    Attributes:
        long_name (str): Full program name
        short_name (str): Program abbreviation
        base_year (int): Base year for cost calculations (1970-2060)
        dol_units (int): Currency unit multiplier (1, 1K, 1M, 1B)
        required_fields (List[str]): Required fields in cost estimates
        report_fields (List[str]): Fields to include in reports
    """
    
    long_name = param.String("Program X")
    short_name = param.String("X")
    base_year = param.Integer(2020, bounds=(1970, 2060))
    dol_units = param.Selector([1, 1_000, 1_000_000, 1_000_000_000], default=1_000)
    required_fields = param.List(['FY', 'value_cp'])
    report_fields = param.List(['model', 'appn', 'FY', 'value_cp', 'value_ty', 'value_cy'])

    @property
    def BY(self) -> int:
        """Return the base year for calculations."""
        return self.base_year
    
    def __panel__(self) -> Any:
        """Create a Panel interface for the parameters."""
        return pn.Column(self.param)


class Model(param.Parameterized): #
    """
    Base class for cost estimation models.
    
    This class provides the foundation for all cost estimation models,
    implementing reactive parameter management and calculation updates.
    
    Attributes:
        meta (Dict): Metadata about the model (analyst, element, etc.)
        global_inputs (GlobalInputs): Program-wide parameters
        uncertainty_inputs (Dict): Uncertainty parameters
        uncertainty (float): Uncertainty factor
        sim_results (pd.DataFrame): Simulation results
        cost_estimate (pd.DataFrame): The calculated cost estimate
        schedule_estimate (pd.DataFrame): The calculated schedule estimate
        total_cost_cp (float): Total constant price cost
        total_cost_ty (float): Total then-year cost
    """
    
    meta = param.Dict(
        default={
            'Analyst': "N/A",
            'Element': "N/A",
        }
    )
    global_inputs = param.ClassSelector(GlobalInputs, GlobalInputs(), instantiate=False)
    uncertainty_inputs = param.Dict(default={
                'uncertainty': sim_engine.RV(mean=1, cv=0.25)
            })
    uncertainty = param.Number(1)
    simulate = param.Action(lambda x: x.run_simulation(100))
    sim_results = param.DataFrame(precedence=.1)
    cost_estimate = param.DataFrame(precedence=.1)
    schedule_estimate = param.DataFrame(precedence=.1)
    total_cost_cp = param.Number(precedence=.1, constant=True)
    total_cost_ty = param.Number(precedence=.1, constant=True)
    
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._cost_estimate = pd.DataFrame()
        self._schedule_estimate = pd.DataFrame()
        # check for cycles in the parameter dependencies
        G = build_param_dependency_graph(self)
        self._graph = G
        self._cycles = check_param_cycles(G)
        if self._cycles:
            logger.warning("Cycles in parameter dependencies")
            # print out the cycles
            for cycle in self._cycles:
                logger.warning(f"Cycle: {cycle}")

            raise ValueError("Cycles in parameter dependencies")
        
    def __call__(self, update: bool = True, **params: Any) -> pd.DataFrame:
        """
        Execute the model calculations.
        
        Args:
            update: Whether to update parameters before calculation
            **params: Parameters to update
            
        Returns:
            The cost estimate DataFrame
        """
        if update:
            self.param.update(**params)
        else:
            logger.warning("Temporary update not implemented yet")
        return self.cost_estimate

    @property
    def parent_path(self) -> str:
        """Get the parent path of the model."""
        if hasattr(self, "_parent_path"):
            return self._parent_path
        return ""
    
    @parent_path.setter
    def parent_path(self, value: str) -> None:
        """Set the parent path of the model."""
        self._parent_path = value

    @property
    def level(self) -> int:
        """Get the model's hierarchy level."""
        level = self.parent_path.count('>') + 1
        return level

    def calc_cost(self) -> pd.DataFrame:
        """
        Calculate the base cost estimate.
        
        This method should be implemented by subclasses to provide
        specific cost calculation logic.
        
        Returns:
            DataFrame containing the cost estimate
        """
        logger.warning("calc_cost is not implemented")
        self.cost_estimate = pd.DataFrame(columns=self.global_inputs.required_fields)
        return self.cost_estimate
    
    @param.depends('calc_cost', 'uncertainty', watch=True)
    def calc_cost_uncertainty(self) -> pd.DataFrame:
        """
        Apply uncertainty factors to the cost estimate.
        
        Returns:
            DataFrame with uncertainty-adjusted costs
        """
        logger.debug("Applying uncertainty factors")
        self._cost_estimate = self.cost_estimate.assign(
            uncertainty=self.uncertainty,
            value_cp=lambda x: x.value_cp * self.uncertainty
        )
        return self._cost_estimate
    
    @param.depends('calc_cost_uncertainty','global_inputs', watch=True)
    def calc_cost_metadata(self) -> pd.DataFrame:
        """
        Create the final cost estimate with model information.
        
        Returns:
            DataFrame containing the final cost estimate
        """
        df = self._cost_estimate.copy()
        # initialize to required fields
        col_order =[]# set(self.global_inputs.required_fields)

        logger.debug("Creating final cost estimate")
        # add required fields to the cost estimate
        missing_fields = [col for col in self.global_inputs.required_fields+self.global_inputs.report_fields if col not in df.columns]
        if len(missing_fields) > 0:
            logger.debug(f"Missing fields: {missing_fields}")
            df = df.assign(
                **{field: None for field in missing_fields}
            )
        # add parent path and level to the cost estimate
        path_list = self.parent_path.split(' > ')
        for i, path in enumerate(path_list):
            col_order.append("model_level_" + str(i))
            df = df.assign(
                **{"model_level_" + str(i): path}
            )
        # add this model's name
        col_order.append("model")
        df = df.assign(
            **{"model": self.__class__.__name__}
        )
        # add cp and ty columns
        col_order.append("value_cp")
        col_order.append("value_ty")
        col_order.append("value_cy")
        col_order.append("global_inputs_base_year")
        col_order.append("global_inputs_dol_units")
        col_order.append("global_inputs_long_name")
        col_order.append("global_inputs_short_name")
        if "cost_units" not in df.columns:
            df = df.assign(
                cost_units=lambda x: "CP"+self.global_inputs.base_year+"$" + ("M" if self.global_inputs.dol_units == 1_000_000 else "B" if self.global_inputs.dol_units == 1_000_000_000 else "")
            )
        col_order.append("cost_units")
        df = df.assign(
            value_cp=lambda x: x.value_cp * self.global_inputs.dol_units, # TODO: add unit conversion functions
            value_ty=lambda x: x.value_cp , # TODO: add escalation functions
            value_cy=lambda x: x.value_ty # TODO: add inflation functions
            **{"global_inputs_" + field: getattr(self.global_inputs, field) for field in ['base_year', 'dol_units', 'long_name', 'short_name']}  
        )
        # get meta columns
        for key, val in self.meta.items():
            col_order.append("meta_" + key)
            df = df.assign(
                **{"meta_" + key: val}
            )
        
        # reorder columns
        df = df[list(col_order)]
        self._cost_estimate = df

        return self._cost_estimate
    
    @property
    def total_cost_cp(self) -> float:
        """
        Get the total cost in constant price.
        """
        if self._total_cost_cp:
            return self._total_cost_cp
        try:
            if "value_cp" not in self._cost_estimate.columns:
                return 0
            else:
                self._total_cost_cp = self._cost_estimate.value_cp.sum()
                return self._total_cost_cp
        except Exception as e:
            logger.error(f"Error calculating total cost in constant price: {e}")
            return 0
    
    @property
    def total_cost_ty(self) -> float:
        """
        Get the total cost in then-year.
        """
        if self._total_cost_ty:
            return self._total_cost_ty
        try:
            if "value_ty" not in self._cost_estimate.columns:
                return 0
            else:
                self._total_cost_ty = self._cost_estimate.value_ty.sum()
                return self._total_cost_ty
        except Exception as e:
            logger.error(f"Error calculating total cost in then-year: {e}")
            return 0
    
    @property
    def total_cost_cy(self) -> float:
        """
        Get the total cost in current year.
        """
        if self._total_cost_cy:
            return self._total_cost_cy
        try:
            if "value_cy" not in self._cost_estimate.columns:
                return 0
            else:
                self._total_cost_cy = self._cost_estimate.value_cy.sum()
                return self._total_cost_cy
        except Exception as e:
            logger.error(f"Error calculating total cost in current year: {e}")
            return 0

