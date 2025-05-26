"""
DEPRECATED: This module has been refactored and split into smaller, more maintainable files.

Please use the individual modules instead:
- pycost.cost_estimate.base: GlobalInputs and Model base classes
- pycost.cost_estimate.parent_models: ParentModel class
- pycost.cost_estimate.model_types: Specialized model implementations
- pycost.cost_estimate.inventory: Inventory management
- pycost.cost_estimate.ui: User interface components

This file is kept for backward compatibility but will be removed in a future version.
"""

import warnings

warnings.warn(
    "The cost_model module is deprecated and will be removed in a future version. "
    "Please use the individual modules instead.",
    DeprecationWarning,
    stacklevel=2
)

# Original content follows
"""
Cost estimation framework for program analysis.

This module provides a framework for developing cost estimates using Python,
offering an alternative to traditional Excel-based cost estimation. It implements
a reactive programming model that automatically updates calculations when inputs change.

Key Components:
- GlobalInputs: Program-wide parameters and settings
- Model: Base class for cost estimation models
- ParentModel: Container for combining multiple cost models
- Various specialized models (Development, Production, LearningCurve)
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
import numbergen as ng
import scipy.stats
import networkx as nx
from .reactive import Reactive
import pycost.cost.sim_tool as sim

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

class Model(param.Parameterized):
    """
    Base class for cost estimation models.
    
    This class provides the foundation for all cost estimation models,
    implementing reactive parameter management and calculation updates.
    
    Attributes:
        meta (Dict): Metadata about the model (analyst, element, etc.)
        GlobalInputs (GlobalInputs): Program-wide parameters
        u_inputs (Dict): Uncertainty parameters
        uncertainty (float): Uncertainty factor
        sim_results (pd.DataFrame): Simulation results
        cost_estimate (pd.DataFrame): The calculated cost estimate
        schedule_estimate (pd.DataFrame): The calculated schedule estimate
    """
    
    meta = param.Dict(
        default={
            'Analyst': "N/A",
            'Element': "N/A",
        }
    )
    global_inputs = param.ClassSelector(GlobalInputs, GlobalInputs(), instantiate=False)
    uncertainty_inputs = param.Dict(
        default={
            'uncertainty': ng.NormalRandom(mu=1, sigma=.25)
        }
    )
    uncertainty = param.Number(1)
    simulate = param.Action(lambda self: self.run_simulation(100))
    sim_results = param.DataFrame(precedence=.1)
    cost_estimate = param.DataFrame(precedence=.1)
    schedule_estimate = param.DataFrame(precedence=.1)
    total_cost_cp = param.Number(precedence=.1, constant=True)
    total_cost_ty = param.Number(precedence=.1, constant=True)
    
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
        self.cost_estimate = self.cost_estimate.assign(
            uncertainty=self.uncertainty,
            value_cp=lambda x: x.value_cp * self.uncertainty
        )
        return self.cost_estimate
    
    @param.depends('calc_cost_uncertainty','global_inputs', watch=True)
    def calc_cost_metadata(self) -> pd.DataFrame:
        """
        Create the final cost estimate with model information.
        
        Returns:
            DataFrame containing the final cost estimate
        """
        df = self.cost_estimate.copy()
        # initialize to required fields
        col_order = set(self.global_inputs.required_fields)

        logger.debug("Creating final cost estimate")
        # add required fields to the cost estimate
        missing_fields = col_order - set(df.columns)
        if len(missing_fields) > 0:
            logger.debug(f"Missing fields: {missing_fields}")
            df = df.assign(
                **{field: None for field in missing_fields}
            )
        # add parent path and level to the cost estimate
        path_list = self.parent_path.split(' > ')
        for i, path in enumerate(path_list):
            col_order.add("Level " + str(i))
            df = df.assign(
                **{"Level " + str(i): path}
            )
        # add this model's name
        col_order.add("Model")
        df = df.assign(
            **{"Model": self.__class__.__name__}
        )
        # add cp and ty columns
        col_order.add("value_cp")
        col_order.add("value_ty")
        df = df.assign(
            value_cp=lambda x: x.value_cp, # TODO: add unit conversion functions
            value_ty=lambda x: x.value_cp * self.global_inputs.dol_units, # TODO: add escalation functions
            value_cy=lambda x: x.value_ty * self.global_inputs.dol_units # TODO: add inflation functions
        )
        # get meta columns
        for key, val in self.meta.items():
            col_order.add("meta_" + key)
            df = df.assign(
                **{"meta_" + key: val}
            )
        # reorder columns
        df = df[list(col_order)]
        self.cost_estimate = df



        return self.cost_estimate

    @param.depends('cost_estimate', watch=True)
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
    
    @param.depends('cost_estimate', watch=True)
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
        self.calc_cost_estimate()
    
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
        import multiprocessing
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
            cost = model.cost_estimate
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
            model: Model object to get
        """
        if i:
            return self.models[i]
        elif model_name:
            for model in self.models:
                if model.name == model_name:
                    return model
            raise ValueError(f"Model {model_name} not found")
        else:
            raise ValueError("No model specified")

    def _add_model(self, model: Model) -> None:
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


class ModelApp(param.Parameterized):
    """
    Application interface for cost estimation models.
    
    This class provides a user interface for interacting with cost models,
    including visualization and parameter adjustment.
    
    Attributes:
        model (Model): The currently selected model
        models (List[Model]): Available models
        model_choice (int): Index of the selected model
    """
    
    model = param.ClassSelector(Model)
    models = param.List()
    model_choice = param.Integer(0)

    @param.depends('model', 'models', 'model_choice')
    def view_available_models(self) -> Any:
        """
        Create a model selection interface.
        
        Returns:
            Panel widget for model selection
        """
        selector = pn.widgets.Select(
            name="Choose Model",
            options={m.name: i for i, m in enumerate(self.models)},
            value=self.model_choice
        )
        if len(self.models) == 0:
            self.models.append(self.model)
        self.model = self.models[self.model_choice]
        return pn.Row(selector)

    @param.depends('model.cost_estimate')
    def view_summary(self) -> Any:
        """
        Create a summary view of the cost estimate.
        
        Returns:
            Panel layout with summary data and plot
        """
        data = self.model.cost_estimate.pivot_table(
            columns="FY",
            values='value_cp',
            aggfunc='sum'
        )
        plt = data.plot(kind='bar')
        return pn.Column(data, pn.Card(plt, title="Plot", sizing_mode='stretch_width'))
    
    @param.depends('model.param')
    def view_outputs(self) -> Any:
        """
        Create a view of the model outputs.
        
        Returns:
            Panel widget with output data
        """
        return pn.widgets.Tabulator(self.model.cost_estimate, header_filters=True)

    def view_model(self) -> Any:
        """
        Create a view of the model inputs and outputs.
        
        Returns:
            Panel layout with model interface
        """
        inputs = []
        for p in self.model.param:
            if getattr(getattr(self.model, p, None), "__panel__", None):
                logger.debug(f"Adding panel for parameter: {p}")
                inputs.append(
                    pn.Card(
                        getattr(getattr(self.model, p), "__panel__"),
                        title=p,
                        collapsed=True
                    )
                )
            else:
                logger.debug(f"Adding widget for parameter: {p}")
                inputs.append(self.model.param[p])
        
        return pn.Column(
            pn.Card(*inputs, title="Inputs"),
            pn.Card(self.view_outputs, title="Outputs", sizing_mode='stretch_width'),
            sizing_mode='stretch_width'
        )
    
    def view_documentation(self) -> str:
        """
        Get the model's documentation.
        
        Returns:
            The model's docstring
        """
        return self.model.__doc__ or "No documentation available"
    
    def view_graph(self) -> Any:
        """
        Create a visualization of the model's dependency graph.
        
        Returns:
            Panel layout with graph visualization
        """
        logger.debug("Generating dependency graph")
        g = Reactive.build_dtree(self.model)
        self.__G__ = g
        fig = Reactive.ShowTree(g, lib="matplotlib")
        df = pd.DataFrame.from_records([{"from": e[0], "to": e[1]} for e in g.edges])
        return pn.Row(pn.pane.Matplotlib(fig), df)
    
    def __find_nested_params(self) -> List[str]:
        """
        Find nested parameterized objects in the model.
        
        Returns:
            List of parameter names that are Parameterized objects
        """
        p_list = []
        for p in self.model.param:
            if isinstance(self.model.param[p], param.Parameterized):
                logger.debug(f"Found nested parameter: {p}")
                p_list.append(p)
        return p_list

    @param.depends('model')
    def __panel__(self) -> Any:
        """
        Create the main application interface.
        
        Returns:
            Panel layout with all interface components
        """
        summary = pn.layout.FloatPanel(
            self.view_summary,
            sizing_mode='stretch_both',
            position='center-top',
            offsety=40,
            offsetx=20,
            contained=False,
            name='Summary: ' + self.model.name,
            config={"headerControls": {"maximize": "remove", "close": "remove"}}
        )
        
        return pn.Column(
            summary,
            self.view_available_models,
            pn.Tabs(
                ('Model', self.view_model),
                ("CEMM", "CEMM"),
                ("Documentation", self.view_documentation),
                ("Graph", self.view_graph),
                sizing_mode='stretch_width'
            ),
            sizing_mode='stretch_width'
        )


class Inventory(param.Parameterized):
    """
    Class for managing program inventory and delivery schedules.
    
    Attributes:
        profile (pd.DataFrame): Inventory profile data
        delivery_cycle (int): Time between procurement and delivery
        service_life (int): Expected service life of items
    """
    
    profile = param.DataFrame(
        default=pd.DataFrame({
            'FY': range(2030, 2040),
            'quantity': [5]*5 + [10]*5
        })
    )
    delivery_cycle = param.Integer(2)
    service_life = param.Integer(20)

    @property
    def procurement(self) -> pd.DataFrame:
        """Get the procurement profile."""
        return self.profile

    @property
    def delivery(self) -> pd.DataFrame:
        """Get the delivery schedule."""
        return self.procurement.FY + self.delivery_cycle

    @property
    def inventory(self) -> pd.DataFrame:
        """Get the inventory profile."""
        return self.delivery
    
    def __panel__(self) -> Any:
        """Create a panel interface for the inventory data."""
        return pn.Card(self.param.profile, title="Inventory", sizing_mode="stretch_width")


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


class Production(Model):
    """
    Model for production phase costs using learning curve analysis.
    
    Attributes:
        first_unit_cost (float): First unit cost (previously T1)
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


class Demo_Program(ParentModel):
    """
    Example program combining development and production models.
    
    This class demonstrates how to combine multiple cost models
    into a complete program estimate with both development and
    production phases.
    
    Attributes:
        inventory (Inventory): Shared inventory object used across models
    """
    
    def __init__(self, inventory=None):
        """
        Initialize the demo program with development and production models.
        
        Args:
            inventory (Inventory, optional): Inventory profile for production.
                If None, a default Inventory instance will be created.
        """
        # Set up the inventory object
        self.inventory = inventory if inventory is not None else Inventory()
        
        # Create the component models
        self.models = [
            Development(
                cost=500, 
                duration=10, 
                start_year=2020,
                meta={
                    'Analyst': 'Demo User',
                    'Element': 'Development',
                }
            ),
            Production(
                inventory=self.inventory,
                meta={
                    'Analyst': 'Demo User',
                    'Element': 'Production',
                }
            )
        ]
        
        # Initialize the parent model
        super(Demo_Program, self).__init__()


class QuantityInputs(param.Parameterized):
    """
    Class for managing quantity-based inputs and schedules.
    
    Attributes:
        procurement (pd.DataFrame): Procurement schedule
        delivery_cycle (int): Time between procurement and delivery
        service_life (int): Expected service life
        delivery (pd.DataFrame): Delivery schedule
        retirement (pd.DataFrame): Retirement schedule
        inventory (pd.DataFrame): Combined inventory profile
    """
    
    procurement = param.DataFrame(
        pd.DataFrame(columns=["FY", "Value"]),
        columns=set(["FY", "Value"])
    )
    delivery_cycle = param.Integer(2)
    service_life = param.Integer(20)
    delivery = param.DataFrame()
    retirement = param.DataFrame()
    inventory = param.DataFrame()
    
    def __init__(self, **params):
        """Initialize the quantity inputs."""
        super(QuantityInputs, self).__init__(**params)
        if not self.procurement.empty:
            self._calc_inventory()

    @param.depends('procurement', 'delivery_cycle', 'service_life', watch=True)
    def _calc_inventory(self) -> None:
        """
        Calculate inventory profiles from procurement data.
        
        This method creates delivery and retirement schedules
        based on the procurement profile and timing parameters.
        """
        logger.debug("Calculating inventory profiles")
        self.delivery = self.procurement.assign(
            FY=self.procurement.FY + self.delivery_cycle
        )
        self.retirement = self.delivery.assign(
            FY=self.delivery.FY + self.service_life
        )
        self.inventory = pd.concat([
            self.procurement.assign(Procurement=lambda x: x.Value).drop('Value', axis=1),
            self.delivery.assign(Delivery=lambda x: x.Value).drop(['Program', 'Value'], axis=1),
            self.retirement.assign(Retirement=lambda x: x.Value).drop(['Program', 'Value'], axis=1)
        ], axis=1)
        self.inventory[list(range(2020, 2050))] = np.nan


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


class LearningCurve(Model):
    """
    Model for manufacturing cost estimation using learning curves.
    
    Attributes:
        first_unit_cost (float): First unit cost (previously T1)
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
            Value=lambda x: self.first_unit_cost * (x.Qty**(np.log(self.learning_curve))) * (x.Rate**(np.log(self.rate_curve)))
        )
        self.cost_estimate = tmp


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


if __name__ == "__main__":
    app = ModelApp(model=Demo_Program())
    app.show()
