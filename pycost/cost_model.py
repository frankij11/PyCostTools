import pandas as pd
import numpy as np
import param
import numbergen as ng
import scipy.stats


import networkx as nx
class Reactive:
    '''
    Base Class to make any Class Object *react* to changes in attributes or chained functions.
    
    Algorithm is basic in that it uses a simple heuritistic to determine dependnces
        1. reads all functions and variables into a list
        2. For each attribute check if it is in the script of any callables
            If it is in the script check if it is being assigned
    
    overwritten methods:
    __setattr__ : intervene in 
    
    new methods:
    _build_dependencies(): create directed graph of calculations
    ShowTree(lib="HV"): draw network of current dependencies
    
    new params:
    __depends__ : list of attributes that are used in a callable
    __preced__ : list of attributes that are set in a callable
    
    Optional:
    __log__ : if you want to enable logging of changes create a class attribute names "__log__" 
    
    
    Example:
    
    class LaborCost(Reactive):
        
        def __init__(self, hours,labor_rate):
            # initialize parameters
            self.hours = hours 
            self.labor_rate = labor_rate
            
            # full calc
            self.__call__()
            
            # This method builds the model relationships
            # This method creates two new dunder variables
                # __depends__
                # __preced__
            self._build_dependencies()
            
            # Add logging to class
            # Can be deleted without affecting model and then recreated
            self.__log__ =[]
        
        def __call__(self):
            self.calc1()
            self.calc2()
            
            return self.labor_cost
            
        def calc1(self):
            self.total_cost = self.hours * self.labor_rate
        
        def calc2(self):
            self.total_cost = self.total_cost * 100
    
    '''
    def __setattr__(self, key,value):
        
        self.__preset__(key,value)
        self.__dict__[key]= value
        self.__postset__(key)
    
    def __preset__(self, key, value):
        
        #update
        try:
            
            if value == self.__dict__[key]:
                pass
            else:
                self.__log__.append(
                    {key:dict(
                            old=self.__dict__[key],
                            new=value)
                    })
        except:
            pass
            #print("First Time", key, "is being set")

    
    def __postset_batch__(self, keys):
        pass
    
    def __postset__(self,key, batch=False,auto_calc=True):
        
        try:
            for func in self.__dtree__[key]:
                getattr(self, func)()
        except:
            pass
            #print("dtree not initialized")

    
    def __call__(self):
        
        for key,item in self.__depends__:
            #print(key)
            self.__postset__(key)
        
        return self
        


    
    def __build_dtree__(self):
        import inspect
        import re
        ops = ["+", "-", "*", "/", "%", "**", "//"]
        asgn = ["=","+=", "-=", "*=", "/=", "%=", "//=", "**=", "&=", "|=", "^=", ">>=", "<<="]
        def check_asgn(var_str, src):
            for s in asgn:
                search = str(var_str) +str(s)
                if search in src:
                    return True
            return False
        
        attributes = inspect.getmembers(self)
        scripts = inspect.getmembers(self, lambda a:inspect.isroutine(a))
        attributes = [a[0] for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
        scripts = [a for a in scripts if not(a[0].startswith('__') and a[0].endswith('__'))]
        from collections import defaultdict
        self.__dtree__ = defaultdict(list)
        self.__depends__ = []
        self.__preced__ = []
        for s in scripts:
            if s[0] != "_script":
                src = inspect.getsource(s[1]).replace(" ","")
                for a in attributes + [s[0] for s in scripts]:
                    #if "self."+a in re.split('\\+|\\-|\\==|\\*|\n', src):
                    if "self."+a in src and not check_asgn("self."+a,src):
                        self.__dtree__[a].append(s[0])
                        self.__depends__.append((a, s[0]))
                    elif "self."+a in src or check_asgn("self."+a,src):
                        self.__preced__.append((s[0],a))
        
        # create networkx graph
        #try:
        import networkx as nx
        G = nx.MultiDiGraph()
        G.add_nodes_from([(s[0], dict(name=s,kind="Func", shape='box',color='blue')) for s in scripts] )
        G.add_nodes_from([(v, dict(name=v, kind="Var", color='orange')) for v in attributes if v not in G.nodes()])
        G.add_edges_from(self.__depends__)
        G.add_edges_from(self.__preced__)
        self.__G__ = G
        #except:
            # networks could not be loaded
            #pass


        
        
    def ShowTree(self, lib ='HV'):
        import networkx as nx
        import inspect
        G = self.__G__
        
        # reverse graph used to calculate depth of node
        H = nx.MultiDiGraph()
        H.add_nodes_from(G)
        H.add_edges_from([(e[1], e[0]) for e in G.edges])
        for n in H.nodes():
            lev = set([val[1] for val in  dict(nx.bfs_predecessors(H, n)).items()])
            lev = len(lev)
            G.nodes[n]['depth'] = lev
        
        h = {i:0 for i in range(100) }
        for n in G.nodes():
            G.nodes[n]['height'] =  h[G.nodes[n]['depth']] *1.5
            G.nodes[n]['pos'] =  (float(G.nodes[n]['depth']), float(G.nodes[n]['height']))
            h[G.nodes[n]['depth']] = h[G.nodes[n]['depth']] +1
        
        
        
        if lib.lower() == 'matplotlib':
            import matplotlib.pyplot as plt 
            plt.figure(figsize=(6,6))
            #pos= [key for key in nx.get_node_attributes(G,'pos').keys()] #  nx.spring_layout(G,scale=2)
            pos= nx.get_node_attributes(G,'pos') #  nx.spring_layout(G,scale=2)
            
            color_map = [G.nodes[g]['color'] for g in G.nodes] 
            nx.draw(G,pos,node_color=color_map,with_labels=True, node_size=1000,connectionstyle='arc3, rad = 0.1')
        if lib.lower() == 'hv':
            import holoviews as hv
            hv.extension('bokeh')
            graph = hv.Graph.from_networkx(G, nx.layout.fruchterman_reingold_layout).opts(
                width=800, height=400,xaxis=None, yaxis=None,legend_position='top_left',
                directed=True,node_size=50,inspection_policy='edges',arrowhead_length=0.01, node_color='color')
            labels = hv.Labels(graph.nodes, ['x','y'], 'name')
            return graph*labels





class GlobalInputs(param.Parameterized):
    BY = param.Integer(2020)
    


class Model(param.Parameterized):
    meta = param.Dict(
        default = {'Analyst': "N/A",
                  'Element': "N/A",
                  }
    )
    g_inputs = param.ClassSelector(GlobalInputs, GlobalInputs(), instantiate=False)
    u_inputs = param.Dict(
        default= {
            'uncertainty': ng.NormalRandom(mu=1,sigma=.25)
            } )
    uncertainty = param.Number(1)
    simulate = param.Action( lambda self: self.run_simulation(100))
    results = param.DataFrame(
        default = pd.DataFrame(columns =['Element','APPN', 'BaseYear', 'FY','Value']),
        doc = '''Results container shared by all estimates. This value should be updated using the generic _calc method''',
        columns = set(['Element', 'APPN', 'BaseYear', 'FY','Value']),
        precedence=.1
    )
    sim_results = param.DataFrame(precedence=.1)



    def __init__(self, **params):
        super().__init__(**params)
        # Automatically set results to a calculation given defaults
        self._calc()


    def _calc(self):
        print(self.name, "Not Implement")
        self.results = self.results
    
    @param.depends('results', watch=True)
    def _update_results(self):
        self.results['Value'] = self.results['Value'] * self.uncertainty


    def _prepare_sim(self):
        if self.u_inputs is not None: 
            self.param.set_param(**self.u_inputs)
        
    def _end_sim(self):
        if self.u_inputs is not None:
            for key,val in self.u_inputs.items():
                self.param.set_param(**{key: self.param[key].default})
        self._calc()
    
    def run_simulation(self, trials=100,clear_previous_sim=True, agg_results = True, agg_columns=['APPN', 'FY']):
        self._prepare_sim()
        if clear_previous_sim: self.sim_results =pd.DataFrame()
        for i in range(trials):
            self._prepare_sim()
            self._calc()
            if agg_results:
                self.sim_results = self.sim_results.append(self.results.groupby(by=agg_columns )['Value'].sum().reset_index().assign(Trial=i))
            else:
                self.sim_results = self.sim_results.append(self.results.assign(Trial=i))
        self._end_sim()
    def run_simulation_parallel(self, trials=100, agg_results=True, agg_columns=['APPN', 'FY']):
        import multiprocessing
        with multiprocessing.Pool() as pool:
            pool.map(self.run_simulation, range(len(self.models)))
    
    
    def build_panel_app(self):
        self.app = pn.Pane(self)
        
    def build_app(self):
        try:
            import panel
            self._build_panel()
        except:
            try:
                import ipywidets
            except:
                print("No dashboard apps available. Try downloading panel or ipywidgets")
    
class Models(Model):
    
    
    models = param.List()
    
    def _calc(self, run_parallel=False):
        results = []
        if run_parallel:
            import multiprocessing
            with multiprocessing.Pool() as pool:
                pool.map(self._calc_model, range(len(self.models)))
        else:
            
            for model in self.models:
                model._calc()
                results.append(model.results.assign(ModelType = type(model).__name__))
        
            if len(results) >0: self.results = pd.concat(results, ignore_index=True)
        
    def _calc_model(self, i):
        self.models[i]._calc()
    
    def _add_model(self, model):
        #for new_param in model.param
            #self.
            #self.param.watch(self._calc, ['a'], queued=True, precedence=2)
        self.models.append(model)
        self._calc()
    
    def _prepare_sim(self):
        if self.u_inputs is not None:
            self.param.set_param(**self.u_inputs)
        for model in self.models:
            model._prepare_sim()
        
    def _end_sim(self):
        if self.u_inputs is not None:
            for key,val in self.u_inputs.items():
                self.param.set_param(**{key: self.param[key].default})
        
        for model in self.models:
            model._end_sim()
        self._calc()


class BaseModel(param.Parameterized):
    '''
    Base model for building a cost estimate. Similar to an
    empty worksheet. All other models should extend this class
    '''
    meta= param.Dict(dict(
        Analyst = param.String("Uknown"),
        WBS = param.String('Uknown') 
        ))

    inputs = GlobalInputs()

    estimate= param.DataFrame(
        default = pd.DataFrame(columns =["Element", "units","FY", "Value"] ), columns=set(["Element", "units","FY", "Value"]))

    def __init___(self, name, categories, analyst, **kwargs):
        pass

    def fit(self, **kwargs):
        '''
        Implement model
        '''
        return self

    def predict(self, inputs=None):
        '''
        Implement predict
        '''
        # given inputs calcuate estimate
        if inputs is None: inputs = self.inputs
        return self.estimate
    
    def ui(self):
        '''
        Generate a worksheet like document to display estimate
        '''
        app=None
        return app

    def simulate(self):
        '''
        Implement simulation
        '''
        pass
    
class CostModel(BaseModel):
    '''
    Collection of estimates, similar to a workbook in Excel
    '''
    inputs = param.Dict()
    global_inputs = param.Dict(dict(
        ProgramName = param.String("NA"),
        BY = param.Integer(2020, bounds=(1970,2060) ),
        EstimateName= param.String("NA")
        ))
    models=param.Dict()
    

    pass


# %%
class QuantiyInputs(param.Parameterized):
    procurement = param.DataFrame(pd.DataFrame(columns = ["FY", "Value"]), columns = set(["FY", "Value"]))
    delivery_cycle = param.Integer(2)
    service_life = param.Integer(20)
    delivery = param.DataFrame()
    retirement = param.DataFrame()
    inventory = param.DataFrame()
    
    def __init__(self, **params):
        super(QuantiyInputs,self).__init__(**params)
        if not self.procurement.empty:
            self._calcInvenotry()


    @param.depends('procurement', 'delivery_cycle', 'service_life', watch=True)
    def _calcInvenotry(self):
        self.delivery = self.procurement.assign(FY = self.procurement.FY + self.delivery_cycle)
        self.retirement = self.delivery.assign(FY=self.delivery.FY + self.service_life)
        self.inventory  =  pd.concat([
            self.procurement.assign(Procurement= lambda x: x.Value).drop('Value', axis=1),
            self.delivery.assign(Delivery= lambda x: x.Value).drop(['Program','Value'], axis=1),
            self.retirement.assign(Retirement= lambda x: x.Value).drop(['Program','Value'], axis=1)], axis=1)
        self.inventory[list(range(2020, 2050))] = np.nan

# %%
class Development(BaseModel):
    pass

class EVM(BaseModel):
    '''
    Structure to develop an EVM estimate
    '''
    pass

class LearningCurve(Model):
    '''
    Structure to develop manufacturing estimate
    '''
    T1 = param.Number(100)
    LC = param.Number(.95, bounds=(.7,1))
    RC = param.Number(.95, bounds=(.7,1))
    QtyProfile = param.DataFrame(pd.DataFrame(dict(
        FY = [2020]*10 + [2021]*20 + [2022]*15 + [2023] *5,
        Qty = np.arange(50) +1,
        Rate = [10]*10 + [20]*20 + [15]*15 +[5]*5
    )))

    @param.depends('T1', 'LC', 'RC', 'QtyProfile', watch=True)
    def _calc(self):
        n = self.QtyProfile.shape[0]
        tmp = self.QtyProfile.assign(
            Element = ['Learn'] * n,
            APPN = ['APN'] * n,
            BaseYear = [2020] * n
             
        ).assign(
            Value = lambda x: self.T1 * (x.Qty **(np.log(self.LC))) * (x.Rate**(np.log(self.RC)))
        )
        self.results= tmp


class Factor(BaseModel):
    pass


if __name__ =="__main__":
    pass
    #GlobalInputs = Inputs()
    #CostEstimate = CostModel(meta={'Author':"Kevin Joy"})
    #CostEstimate.predict(inputs = GlobalInputs)
# %%
