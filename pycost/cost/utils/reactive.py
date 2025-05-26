"""
Reactive dependency management system for cost estimation models.

This module provides a base class for implementing reactive programming patterns
in cost estimation models. It automatically tracks dependencies between attributes
and methods, ensuring that calculations are updated when their inputs change.

The Reactive class uses a simple heuristic to determine dependencies:
1. Reads all functions and variables into a list
2. For each attribute, checks if it is used in any callable
3. If an attribute is used, checks if it is being assigned
"""

import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import networkx as nx
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Reactive:
    """
    Base class for implementing reactive programming in cost estimation models.
    
    This class provides automatic dependency tracking and calculation updates
    when model parameters change. It uses a directed graph to represent the
    relationships between attributes and methods.
    
    Attributes:
        __depends__ (List[Tuple[str, str]]): List of (attribute, method) pairs
            representing dependencies
        __preced__ (List[Tuple[str, str]]): List of (method, attribute) pairs
            representing precedence relationships
        __dtree__ (Dict[str, List[str]]): Dictionary mapping attributes to
            methods that depend on them
        __log__ (List[Dict]): List of change logs if logging is enabled
        __G__ (nx.MultiDiGraph): NetworkX graph representing the dependency structure
    
    Example:
        >>> class LaborCost(Reactive):
        ...     def __init__(self, hours, labor_rate):
        ...         self.hours = hours
        ...         self.labor_rate = labor_rate
        ...         self.__call__()
        ...         self._build_dependencies()
        ...         self.__log__ = []
        ...     
        ...     def __call__(self):
        ...         self.calc1()
        ...         self.calc2()
        ...         return self.labor_cost
        ...     
        ...     def calc1(self):
        ...         self.total_cost = self.hours * self.labor_rate
        ...     
        ...     def calc2(self):
        ...         self.total_cost = self.total_cost * 100
    """
    
    def __setattr__(self, key: str, value: Any) -> None:
        """
        Override attribute setting to implement reactive behavior.
        
        Args:
            key: The attribute name
            value: The new value to set
            
        This method:
        1. Logs the change if logging is enabled
        2. Updates the attribute value
        3. Triggers dependent calculations
        """
        self.__preset__(key, value)
        self.__dict__[key] = value
        self.__postset__(key)
    
    def __preset__(self, key: str, value: Any) -> None:
        """
        Log attribute changes before they occur.
        
        Args:
            key: The attribute name
            value: The new value to be set
            
        This method is called before an attribute is changed to log the change
        if logging is enabled.
        """
        try:
            if value == self.__dict__[key]:
                logger.debug(f"No change in {key}, skipping update")
                return
                
            if hasattr(self, '__log__'):
                self.__log__.append({
                    key: {
                        'old': self.__dict__[key],
                        'new': value,
                        'timestamp': pd.Timestamp.now()
                    }
                })
                logger.info(f"Attribute {key} changing from {self.__dict__[key]} to {value}")
        except KeyError:
            logger.debug(f"First time setting attribute {key}")
    
    def __postset_batch__(self, keys: List[str]) -> None:
        """
        Process multiple attribute changes after they occur.
        
        Args:
            keys: List of attribute names that were changed
            
        This method can be overridden by subclasses to implement custom
        batch processing of attribute changes.
        """
        pass
    
    def __postset__(self, key: str, batch: bool = False, auto_calc: bool = True) -> None:
        """
        Trigger dependent calculations after an attribute changes.
        
        Args:
            key: The attribute name that changed
            batch: Whether this is part of a batch update
            auto_calc: Whether to automatically trigger calculations
            
        This method:
        1. Looks up dependent methods in the dependency tree
        2. Calls each dependent method to update calculations
        """
        try:
            for func in self.__dtree__[key]:
                logger.debug(f"Triggering {func} due to change in {key}")
                getattr(self, func)()
        except AttributeError:
            logger.debug("Dependency tree not initialized")
    
    def __call__(self) -> Any:
        """
        Execute all calculations in dependency order.
        
        Returns:
            The final calculation result
            
        This method:
        1. Processes all dependencies in order
        2. Returns the final calculation result
        """
        for key, item in self.__depends__:
            logger.debug(f"Processing dependency: {key} -> {item}")
            self.__postset__(key)
        return self
    
    def __build_dtree__(self) -> None:
        """Build the dependency tree for the current instance."""
        Reactive.build_dtree(self)
    
    @staticmethod
    def build_dtree(self) -> nx.MultiDiGraph:
        """
        Build a directed graph representing the dependency structure.
        
        Args:
            self: The instance to analyze
            
        Returns:
            A NetworkX MultiDiGraph representing the dependencies
            
        This method:
        1. Analyzes the class methods and attributes
        2. Identifies dependencies between attributes and methods
        3. Creates a directed graph representation
        """
        import re
        
        # Define operators and assignment patterns
        ops = ["+", "-", "*", "/", "%", "**", "//"]
        asgn = ["=", "+=", "-=", "*=", "/=", "%=", "//=", "**=", "&=", "|=", "^=", ">>=", "<<="]
        
        def check_asgn(var_str: str, src: str) -> bool:
            """Check if a variable is being assigned in source code."""
            for s in asgn:
                if f"{var_str}{s}" in src:
                    return True
            return False
        
        # Get class members
        attributes = inspect.getmembers(self)
        scripts = inspect.getmembers(self, lambda a: inspect.isroutine(a))
        
        # Filter out private members
        attributes = [a[0] for a in attributes if not (a[0].startswith('_') or a[0].endswith('_'))]
        scripts = [a for a in scripts if not (a[0].startswith('_') or a[0].endswith('_'))]
        
        # Initialize dependency tracking
        # Create a graph for visualization
        G = nx.MultiDiGraph()
        
        # Add all attributes and scripts to the graph
        for a in attributes:
            G.add_node(a, type="attribute")
        for s in scripts:
            G.add_node(s[0], type="script")
        
        # Find dependencies between scripts and attributes
        for script_name, script_obj in scripts:
            # Skip methods that don't have source code
            if not inspect.isfunction(script_obj) or script_name == "__init__":
                continue
                
            try:
                # Get the source code of the method
                source = inspect.getsource(script_obj)
                
                # Check if each attribute is used in this method
                for attr in attributes:
                    # Skip attributes starting with "__" (these are private to class)
                    if attr.startswith("__"):
                        continue
                        
                    # Check for various ways the attribute might be used
                    if f"self.{attr}" in source:
                        # If the attribute is being assigned, add a relation
                        if check_asgn(f"self.{attr}", source):
                            logger.debug(f"Found assignment: {script_name} -> {attr}")
                            G.add_edge(script_name, attr, label="assigns")
                        # Otherwise add a dependency relation
                        else:
                            logger.debug(f"Found dependency: {attr} -> {script_name}")
                            G.add_edge(attr, script_name, label="depends")
                            
            except TypeError:
                logger.warning(f"Cannot get source for {script_name}")
        
        # Store the dependency graph
        self.__G__ = G
        
        # Build the dependency tree as a dictionary
        self.__dtree__ = {}
        for n in G.nodes:
            # If this is an attribute, find all methods that depend on it
            if 'type' in G.nodes[n] and G.nodes[n]['type'] == "attribute":
                self.__dtree__[n] = []
                # Add all outgoing edges that are scripts (methods that depend on this attribute)
                for e in G.out_edges(n):
                    if 'type' in G.nodes[e[1]] and G.nodes[e[1]]['type'] == "script":
                        self.__dtree__[n].append(e[1])
                    
        logger.debug(f"Built dependency tree: {self.__dtree__}")
        return G
        
    @staticmethod
    def ShowTree(G: nx.MultiDiGraph, lib: str = 'HV') -> Union[Any, None]:
        """
        Visualize the dependency graph.
        
        Args:
            G: The NetworkX graph to visualize
            lib: The visualization library to use ('HV' for HoloViews or 'matplotlib')
            
        Returns:
            The visualization object
            
        This method:
        1. Determines which visualization library to use
        2. Creates a visualization of the dependency graph
        """
        if lib.lower() == 'hv':
            try:
                import holoviews as hv
                from holoviews import opts
                
                # Create points for the nodes
                nodes = hv.Scatter(
                    [(x, y) for x, y in zip(range(len(G.nodes)), range(len(G.nodes)))],
                    label="Nodes"
                )
                
                # Create segments for the edges
                edges = hv.Curve(
                    [(x, y) for x, y in G.edges],
                    label="Edges"
                )
                
                # Combine the elements
                graph = nodes * edges
                graph.opts(
                    opts.Curve(color='blue', line_width=2),
                    opts.Scatter(color='red', size=10)
                )
                
                return graph
                
            except ImportError:
                logger.warning("HoloViews is not installed for visualization")
                return None
                
        elif lib.lower() == 'matplotlib':
            try:
                import matplotlib.pyplot as plt
                import networkx as nx
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Set node colors based on type
                colors = []
                for node in G.nodes():
                    if 'type' in G.nodes[node]:
                        if G.nodes[node]['type'] == 'attribute':
                            colors.append('lightblue')
                        else:
                            colors.append('lightgreen')
                    else:
                        colors.append('gray')
                
                # Draw the graph
                pos = nx.spring_layout(G, k=0.3, iterations=50)
                nx.draw(
                    G, pos, with_labels=True, node_color=colors,
                    node_size=1500, font_size=10, font_weight='bold',
                    arrowsize=15, ax=ax
                )
                
                return fig
                
            except ImportError:
                logger.warning("Matplotlib is not installed for visualization")
                return None
                
        else:
            logger.warning(f"Unsupported visualization library: {lib}")
            return None 
        
import param
import networkx as nx
import inspect
def assign_levels(G):
    """
    Assigns levels to each node in the graph based on dependencies.

    Parameters:
    - G: A networkx.DiGraph object representing the dependencies.

    Returns:
    - A dictionary mapping each node to its level.
    """
    levels = {}
    for generation, nodes in enumerate(nx.topological_generations(G)):
        for node in nodes:
            levels[node] = generation
    return levels

def build_param_dependency_graph(instance, include_all_attributes: bool = True):
    """
    Constructs a directed graph representing the dependencies
    within a param.Parameterized instance.

    Parameters:
    - instance: An instance of a class inheriting from param.Parameterized.

    Returns:
    - A networkx.DiGraph object representing the dependencies.
    """
    
    # create a graph from Reactive.build_dtree to gather all known attributes and methods
    if include_all_attributes:
        G = Reactive.build_dtree(instance)
    else:
        G = nx.DiGraph()
    cls = instance.__class__
    
    # Add parameter nodes
    for param_name in instance.param:
        G.add_node(param_name, type='parameter')


    # Inspect methods for dependencies
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        dinfo = getattr(method, '_dinfo', None)
        if dinfo and 'dependencies' in dinfo:
            method_name = name
            G.add_node(method_name, type='method')
            for dep in dinfo['dependencies']:
                # Handle nested dependencies like 'other.param'
                dep_name = dep.split('.')[-1]
                G.add_edge(dep_name, method_name)

    # add start node
    G.add_node('start', type='start')
    G.add_edge('start', instance.param.keys()[0])

    return G
import matplotlib.pyplot as plt

def display_param_dependency_graph(G):
    """
    Displays the dependency graph with nodes positioned based on their level.

    Parameters:
    - G: A networkx.DiGraph object representing the dependencies.
    """
    levels = assign_levels(G)

    # Group nodes by level
    level_nodes = {}
    for node, level in levels.items():
        level_nodes.setdefault(level, []).append(node)

    # Assign positions to nodes
    pos = {}
    for level, nodes in level_nodes.items():
        y = 1
        for node in nodes:
            pos[node] = (level, y)
            y += 1

    # Draw nodes with different colors based on their type
    param_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'parameter']
    method_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'method']

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, nodelist=param_nodes, node_color='lightblue', label='Parameters')
    nx.draw_networkx_nodes(G, pos, nodelist=method_nodes, node_color='lightgreen', label='Methods')
    nx.draw_networkx_edges(G, pos, arrows=True)
    nx.draw_networkx_labels(G, pos)

    plt.title("Parameter Dependencies")
    plt.xlabel("Level (Order of Operations)")
    plt.ylabel("Position within Level")
    plt.legend()
    plt.axis('off')
    plt.show()

def check_param_cycles(G):
    """
    Checks for cycles in the parameter dependency graph.

    Parameters:
    - G: A networkx.DiGraph object representing the dependencies.
    """
    try:
        return nx.find_cycle(G, orientation='original')
    except nx.NetworkXNoCycle:
        return None

