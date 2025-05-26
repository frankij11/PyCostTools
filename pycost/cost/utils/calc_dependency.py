import inspect
import re

# Try to import graphviz, but provide fallbacks if not available
try:
    import graphviz
    _has_graphviz = True
except ImportError:
    print("graphviz is not installed")
    _has_graphviz = False

def create_class_graph(cls):
    # Create a Graphviz graph
    if not _has_graphviz:
        print("Cannot create graph: graphviz not installed")
        return None
        
    dot = graphviz.Digraph()

    # Add nodes for each member of the class and its superclasses
    for c in inspect.getmro(cls):
        for name, member in inspect.getmembers(c):
            if not name.startswith("__"):
                dot.node(name)

    # Add edges between members that reference each other
    for c in inspect.getmro(cls):
        for name, member in inspect.getmembers(c):
            if not name.startswith("__"):
                for referenced_name in get_referenced_names(member):
                    dot.edge(name, referenced_name)

    # Add edges between subclasses and their superclasses
    for c in inspect.getmro(cls)[1:]:
        dot.edge(c.__name__, c.__base__.__name__)

    return dot

def get_referenced_names(obj):
    names = set()

    # Add the names of any attributes that are referenced by the object
    if hasattr(obj, "__code__"):
        for name in obj.__code__.co_names:
            names.add(name)

    # Add the names of any methods or attributes that are referenced by the object's docstring
    if obj.__doc__:
        for name in get_names_from_docstring(obj.__doc__):
            names.add(name)

    return names

def get_names_from_docstring(docstring):
    # Extract any words that are preceded by an "at" symbol (@) in the docstring
    return re.findall(r"@(\w+)", docstring)


class CalcDependency:
    """
    Utility class for analyzing calculation dependencies.
    
    This class provides tools for creating and visualizing dependency graphs
    of classes and their members. It can be used to understand the relationships
    between different parts of a cost model.
    """
    
    @staticmethod
    def create_graph(cls):
        """
        Create a dependency graph for a class.
        
        Args:
            cls: The class to analyze
            
        Returns:
            A Graphviz graph representing the dependencies
        """
        return create_class_graph(cls)
    
    @staticmethod
    def visualize_graph(graph, output_file=None, format='png'):
        """
        Visualize a dependency graph.
        
        Args:
            graph: The Graphviz graph to visualize
            output_file: The file to save the visualization to
            format: The format of the output file
            
        Returns:
            The rendered graph
        """
        if not _has_graphviz:
            print("Cannot visualize graph: graphviz not installed")
            return None
            
        if output_file:
            return graph.render(output_file, format=format)
        return graph


# Example classes for demo purposes
class MyBaseClass:
    def base_method(self):
        pass

class MySubClass(MyBaseClass):
    def sub_method(self):
        self.base_method()

# Only run the example if this module is executed directly
if __name__ == "__main__":
    graph = create_class_graph(MySubClass)
    if graph:
        graph.render("my_class_graph.png") 