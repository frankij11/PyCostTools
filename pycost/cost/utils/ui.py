"""
User interface components for cost estimation.

This module provides UI components for interacting with cost models:
- ModelApp: Main application interface for cost estimation models
"""

import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import param
try:
    import panel as pn
    from panel.pane import Matplotlib
except ImportError:
    pn = None
    Matplotlib = None

#from pycost.cost_estimate.core.base import Model
#from pycost.cost_estimate.utils.reactive import Reactive

# Get logger
logger = logging.getLogger(__name__)


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
        if pn is None:
            logger.warning("Panel is not installed, cannot create UI")
            return None
            
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
        if pn is None:
            logger.warning("Panel is not installed, cannot create UI")
            return None
            
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
        if pn is None:
            logger.warning("Panel is not installed, cannot create UI")
            return None
            
        return pn.widgets.Tabulator(self.model.cost_estimate, header_filters=True)

    def view_model(self) -> Any:
        """
        Create a view of the model inputs and outputs.
        
        Returns:
            Panel layout with model interface
        """
        if pn is None:
            logger.warning("Panel is not installed, cannot create UI")
            return None
            
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
        if pn is None or Matplotlib is None:
            logger.warning("Panel or Matplotlib is not installed, cannot create graph visualization")
            return None
            
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
        if pn is None:
            logger.warning("Panel is not installed, cannot create UI")
            return None
            
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
    
    def show(self):
        """
        Display the application interface.
        
        This method serves the Panel application interface.
        It is designed to work in both notebook and standalone environments.
        """
        if pn is None:
            logger.warning("Panel is not installed, cannot show UI")
            return None
            
        return pn.serve(self.__panel__) 