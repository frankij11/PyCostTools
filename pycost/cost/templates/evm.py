"""
Earned Value Management model implementation.

This module provides a model for EVM analysis:
- EVM: Model for Earned Value Management analysis
"""

import logging
import pandas as pd
import param

from pycost.cost.core import Model

# Get logger
logger = logging.getLogger(__name__)


class EVM(Model):
    """
    Model for Earned Value Management analysis.
    
    This class provides a framework for developing EVM-based
    cost estimates and tracking project performance.
    
    Attributes:
        base_model (Model): Base cost model to track
        start_date (str): Project start date
        report_date (str): Current reporting date
        actual_costs (pd.DataFrame): Actual costs incurred to date
        earned_value (pd.DataFrame): Value earned to date
    """
    
    base_model = param.ClassSelector(Model, default=None)
    start_date = param.String(default=None)
    report_date = param.String(default=None)
    actual_costs = param.DataFrame(default=pd.DataFrame())
    earned_value = param.DataFrame(default=pd.DataFrame())
    
    def __init__(self, **params):
        """Initialize the EVM model."""
        super(EVM, self).__init__(**params)
        self.evm_metrics = pd.DataFrame()
        
    def calc_cost(self) -> pd.DataFrame:
        """
        Calculate EVM metrics based on the cost model and actuals.
        
        Returns:
            DataFrame with EVM metrics
        """
        logger.debug("Calculating EVM metrics")
        
        if self.base_model is None:
            logger.warning("No base model provided for EVM analysis")
            self.cost_estimate = pd.DataFrame()
            return self.cost_estimate
            
        # Calculate planned value (PV) from the base model
        self.base_model.calc_cost()
        planned_value = self.base_model.cost_estimate.copy()
        
        # Calculate EVM metrics
        if not self.actual_costs.empty and not self.earned_value.empty:
            # Create a DataFrame to store EVM metrics
            self.evm_metrics = pd.DataFrame({
                'Metric': ['PV', 'EV', 'AC', 'CV', 'SV', 'CPI', 'SPI', 'EAC', 'ETC', 'VAC'],
                'Description': [
                    'Planned Value', 
                    'Earned Value', 
                    'Actual Cost',
                    'Cost Variance', 
                    'Schedule Variance',
                    'Cost Performance Index', 
                    'Schedule Performance Index',
                    'Estimate at Completion', 
                    'Estimate to Complete',
                    'Variance at Completion'
                ]
            })
            
            # Calculate totals
            pv_total = planned_value['value_cp'].sum()
            ev_total = self.earned_value['value'].sum() if 'value' in self.earned_value.columns else 0
            ac_total = self.actual_costs['value'].sum() if 'value' in self.actual_costs.columns else 0
            
            # Calculate basic EVM metrics
            cv = ev_total - ac_total  # Cost Variance
            sv = ev_total - pv_total  # Schedule Variance
            cpi = ev_total / ac_total if ac_total != 0 else 1  # Cost Performance Index
            spi = ev_total / pv_total if pv_total != 0 else 1  # Schedule Performance Index
            bac = planned_value['value_cp'].sum()  # Budget at Completion
            eac = bac / cpi if cpi != 0 else bac  # Estimate at Completion
            etc = eac - ac_total  # Estimate to Complete
            vac = bac - eac  # Variance at Completion
            
            # Add values to the metrics DataFrame
            self.evm_metrics['Value'] = [
                pv_total, ev_total, ac_total, 
                cv, sv, cpi, spi, eac, etc, vac
            ]
            
        # Combine planned and actual data for the cost estimate
        self.cost_estimate = planned_value
        if not self.actual_costs.empty:
            self.cost_estimate = pd.concat([self.cost_estimate, self.actual_costs], ignore_index=True)
            
        return self.cost_estimate
        
    def summary(self) -> pd.DataFrame:
        """
        Get a summary of EVM metrics.
        
        Returns:
            DataFrame with EVM metrics summary
        """
        return self.evm_metrics 