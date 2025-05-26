from pycost.cost.core.base import Model
import param
import pandas as pd

class Labor(Model):
    """
    A simple model with a single parameter.
    """
    hours = param.Number(default=1, bounds=(0, 10))
    rate = param.Number(default=1, bounds=(0, 10))
    
    @param.depends('hours', 'rate')
    def calc_cost(self):
        self.cost_estimate = pd.DataFrame({'hours': [self.hours], 'rate': [self.rate], 'value_cp': [self.hours * self.rate]})
        return self.cost_estimate
    

if __name__ == '__main__':
    labor = Labor()
    from pycost.cost.utils.reactive import Reactive, build_param_dependency_graph, display_param_dependency_graph
    import networkx as nx
    G = Reactive.build_dtree(labor)
    print(G)
    G=build_param_dependency_graph(labor)
    display_param_dependency_graph(G)

    # determine if there is a cycle in the dependency tree
    if nx.is_directed_acyclic_graph(G):
        print("No cycles in the dependency tree")
    else:
        print("There is a cycle in the dependency tree")
    # determine if there is a cycle in param.depends


    #print(check_param_cycles(labor))


    
    #print(labor.calc_cost())
    #print(labor.calc_cost_uncertainty())
    #print(labor.calc_cost_metadata())
    
    

