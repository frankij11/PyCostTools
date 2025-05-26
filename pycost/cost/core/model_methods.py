"""
Model method implementations for cost estimation.

This module contains implementations of Model methods related to:
- Schedule calculations
- Simulation and uncertainty analysis
- Cost totaling
"""

import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import param
import multiprocessing
from functools import partial

#from pycost.cost_estimate.core.base import Model

# Get logger
logger = logging.getLogger(__name__)

class AdditionalMethods:
    """
    Mixin class containing additional methods for the Model class.
    
    This class provides methods for schedule calculations, simulation,
    and other utilities for cost models.
    """
    
    def calc_schedule_estimate(self) -> pd.DataFrame:
        """
        Calculate schedule information from the cost estimate.
        
        This method creates a schedule estimate by finding the
        minimum and maximum fiscal years in the cost estimate.
        
        Returns:
            DataFrame with schedule summary
        """
        logger.debug("Calculating schedule estimate")
        
        # Handle empty cost estimates
        if self.cost_estimate.empty:
            self.schedule_estimate = pd.DataFrame(columns=['start_year', 'end_year'])
            return self.schedule_estimate
            
        # Look for fiscal year column with case-insensitive matching
        fiscal_year_cols = [col for col in self.cost_estimate.columns 
                        if col.lower() in ('fy', 'fiscal_year', 'year')]
        
        if not fiscal_year_cols:
            logger.warning("No fiscal year column found in cost estimate")
            self.schedule_estimate = pd.DataFrame(columns=['start_year', 'end_year'])
            return self.schedule_estimate
            
        # Use the first matching fiscal year column
        fy_col = fiscal_year_cols[0]
        logger.debug(f"Using '{fy_col}' as fiscal year column")
        
        # Filter for non-zero cost entries
        df = self.cost_estimate.query("value_cp > 0")
        
        if df.empty:
            logger.debug("No non-zero cost entries found")
            self.schedule_estimate = pd.DataFrame({'start_year': [None], 'end_year': [None]})
        else:
            self.schedule_estimate = pd.DataFrame({
                'start_year': [df[fy_col].min()],
                'end_year': [df[fy_col].max()]
            })
            
        return self.schedule_estimate

    def calc_total_cost(self) -> pd.DataFrame:
        """
        Calculate the total cost estimate.
        """
        if not self.cost_estimate.empty:
            if 'value_cp' in self.cost_estimate.columns:
                self.total_cost_cp = self.cost_estimate['value_cp'].sum()
            else:
                logger.warning("Total cost calculation failed: 'value_cp' column not found")
                self.total_cost_cp = None
            if 'value_ty' in self.cost_estimate.columns:
                self.total_cost_ty = self.cost_estimate['value_ty'].sum()
            else:
                logger.warning("Total cost calculation failed: 'value_ty' column not found")
                self.total_cost_ty = None
        else:
            self.total_cost_cp = None
            self.total_cost_ty = None

    def _get_param_string(self) -> str:
        """
        Get a string representation of the parameters used in cost calculation.
        Excludes common parameters like total_cost, include, property_reference.
        """
        excluded_params = {'total_cost', 'include', 'property_reference', 'name', 'description', 'cost_type'}
        param_strings = []
        
        for name, param_obj in self.param.objects().items():
            if name not in excluded_params:
                value = getattr(self, name)
                param_strings.append(f"{name}={value}")
                
        return ", ".join(param_strings)

    def _prepare_sim(self) -> None:
        """
        Prepare the model for simulation by updating parameters.
        
        This method applies uncertainty distributions from uncertainty_inputs
        to the model parameters before running simulations.
        
        Raises:
            ValueError: If an uncertainty input doesn't match an existing parameter
            TypeError: If an uncertainty input value is not compatible with the parameter type
        """
        if self.uncertainty_inputs is not None:
            logger.debug("Updating parameters for simulation")
            
            # Validate uncertainty inputs before applying them
            for param_name, param_value in self.uncertainty_inputs.items():
                if param_name not in self.param:
                    logger.warning(f"Parameter '{param_name}' not found in model")
                    continue
                
                try:
                    # Try to update each parameter individually to catch type errors
                    self.param[param_name].validate(param_value)
                except Exception as e:
                    logger.error(f"Error updating parameter '{param_name}': {str(e)}")
                    raise TypeError(f"Invalid value for parameter '{param_name}': {str(e)}")
            
            # Update all parameters at once
            try:
                self.param.update(**self.uncertainty_inputs)
            except Exception as e:
                logger.error(f"Error updating uncertainty parameters: {str(e)}")
                raise ValueError(f"Failed to update uncertainty parameters: {str(e)}")

    def _end_sim(self) -> None:
        """Clean up after simulation by resetting parameters."""
        if self.uncertainty_inputs is not None:
            logger.debug("Resetting parameters after simulation")
            for key, val in self.uncertainty_inputs.items():
                self.param.update(**{key: self.param[key].default})
        self.calc_cost()

    def run_simulation(
        self,
        trials: int = 100,
        clear_previous_sim: bool = True,
        agg_results: bool = True,
        agg_columns: List[str] = ['FY']
    ) -> None:
        """
        Run Monte Carlo simulation of the cost model.
        
        Args:
            trials: Number of simulation trials
            clear_previous_sim: Whether to clear previous results
            agg_results: Whether to aggregate results
            agg_columns: Columns to aggregate by
        """
        logger.info(f"Starting simulation with {trials} trials")
        self._prepare_sim()
        
        if clear_previous_sim:
            self.sim_results = pd.DataFrame()
            
        for i in range(trials):
            logger.debug(f"Running trial {i+1}/{trials}")
            self._prepare_sim()
            self.calc_cost()
            
            if agg_results:
                trial_results = self.cost_estimate.groupby(by=agg_columns)['value_cp'].sum()
                trial_results = trial_results.reset_index().assign(Trial=i)
                self.sim_results = pd.concat([self.sim_results, trial_results], ignore_index=True)
            else:
                self.sim_results = pd.concat(
                    [self.sim_results, self.cost_estimate.assign(Trial=i)],
                    ignore_index=True
                )
                
        self._end_sim()
        logger.info("Simulation completed")

    def run_simulation_parallel(
        self,
        trials: int = 100,
        agg_results: bool = True,
        agg_columns: List[str] = ['APPN', 'FY']
    ) -> None:
        """
        Run parallel Monte Carlo simulation.
        
        Args:
            trials: Number of simulation trials
            agg_results: Whether to aggregate results
            agg_columns: Columns to aggregate by
        """
        from functools import partial
        
        logger.info(f"Starting parallel simulation with {trials} trials")
        
        # Create a partial function with fixed arguments
        run_sim_task = partial(
            self._run_simulation_task, 
            trials_per_worker=trials // multiprocessing.cpu_count(),
            agg_results=agg_results,
            agg_columns=agg_columns
        )
        
        # Create a worker pool and distribute the tasks
        with multiprocessing.Pool() as pool:
            results = pool.map(run_sim_task, range(multiprocessing.cpu_count()))
            
        # Combine the results from all workers
        if results:
            self.sim_results = pd.concat(results, ignore_index=True)
            
        logger.info("Parallel simulation completed")

    def _run_simulation_task(self, worker_id: int, trials_per_worker: int, 
                        agg_results: bool, agg_columns: List[str]) -> pd.DataFrame:
        """
        Execute a batch of simulation trials for a worker thread.
        
        Args:
            worker_id: ID of the worker thread
            trials_per_worker: Number of trials for this worker
            agg_results: Whether to aggregate results
            agg_columns: Columns to aggregate by
            
        Returns:
            DataFrame with simulation results for this worker
        """
        logger.debug(f"Worker {worker_id} starting with {trials_per_worker} trials")
        
        results = pd.DataFrame()
        
        for i in range(trials_per_worker):
            self._prepare_sim()
            self.calc_cost()
            
            if agg_results:
                trial_results = self.cost_estimate.groupby(by=agg_columns)['value_cp'].sum()
                trial_results = trial_results.reset_index().assign(Trial=worker_id * trials_per_worker + i)
                results = pd.concat([results, trial_results], ignore_index=True)
            else:
                results = pd.concat(
                    [results, self.cost_estimate.assign(Trial=worker_id * trials_per_worker + i)],
                    ignore_index=True
                )
                
        self._end_sim()
        logger.debug(f"Worker {worker_id} completed {trials_per_worker} trials")
        
        return results