# %%
import pandas as pd
import numpy as np
import param
try:
    import panel as pn
except:
    pass
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
        Reactive.build_dtree(self)

    @staticmethod    
    def build_dtree(self):
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
        attributes = [a[0] for a in attributes if not(a[0].startswith('_') or a[0].endswith('_'))]
        scripts = [a for a in scripts if not(a[0].startswith('_') or a[0].endswith('_'))]
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
        no_edges = nx.isolates(G)
        G.add_node('start', kind='Var', shape='diamond', color='white')
        for n in no_edges:
            G.add_edge("start", n)

        self.__G__ = G
        
        #except:
            # networks could not be loaded
            #pass
        return G


        
    @staticmethod    
    def ShowTree(G, lib ='HV'):
        import networkx as nx
        import inspect
        #G = self.__G__
        
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
            fig, ax = plt.subplots()
            #pos= [key for key in nx.get_node_attributes(G,'pos').keys()] #  nx.spring_layout(G,scale=2)
            pos= nx.get_node_attributes(G,'pos') #  nx.spring_layout(G,scale=2)
            
            color_map = [G.nodes[g]['color'] for g in G.nodes] 
            nx.draw(G,pos,node_color=color_map,with_labels=True, node_size=1000,connectionstyle='arc3, rad = 0.1', ax=ax)
            return fig
        if lib.lower() == 'hv':
            import holoviews as hv
            hv.extension('bokeh')
            graph = hv.Graph.from_networkx(G, nx.layout.fruchterman_reingold_layout).opts(
                width=800, height=400,xaxis=None, yaxis=None,legend_position='top_left',
                directed=True,node_size=50,inspection_policy='edges',arrowhead_length=0.01, node_color='color')
            labels = hv.Labels(graph.nodes, ['x','y'], 'name')
            return graph*labels





class GlobalInputs(param.Parameterized):
    long_name = param.String("Program X")
    short_name = param.String("X")
    base_year = param.Integer(2020, bounds=(1970, 2060))
    dol_units = param.Selector([1, 1_000, 1_000_000, 1_000_000_000],default = 1_000)
    required_fields = param.List(['FY','value_cp'])
    report_fields = param.List(['model','appn', 'FY','value_cp','value_ty', 'value_cy']) 

    @property
    def BY(self):
        return self.base_year
    
    def __panel__(self):
        return pn.Column(self.param)

class Model(param.Parameterized):
    meta = param.Dict(
        default = {'Analyst': "N/A",
                  'Element': "N/A",
                  }
    )
    GlobalInputs = param.ClassSelector(GlobalInputs, GlobalInputs(), instantiate=False)
    u_inputs = param.Dict(
        default= {
            'uncertainty': ng.NormalRandom(mu=1,sigma=.25)
            } )
    uncertainty = param.Number(1)
    simulate = param.Action( lambda self: self.run_simulation(100))
    #results = param.DataFrame()
    sim_results = param.DataFrame(precedence=.1)
    
    def __call__(self,update=True, **params):
        if update:
            self.param.update(**params)
        else:
            #store orignial
            print("temporrary update not implemented yet")
        return self.cost_estimate


    @property
    def level(self):
        if hasattr(self,"_level"):
            return self._level
        else:
            self._level = 1
            return 1
    @level.setter
    def level(self,value):
        self._level = value


    def calc_cost(self):
        print("calc_cost is not implemented error")
        self.results = pd.DataFrame(columns = self.GlobalInputs.required_fields)
        return self.results
    
    @param.depends('calc_cost','uncertainty', watch=True)
    def calc_cost_uncertainty(self):
        
        self.results_uncertainty = self.results.assign(value_cp= lambda x: x.value_cp * self.uncertainty)
        return self.results_uncertainty
    
    
    @param.depends('calc_cost_uncertainty', watch=True)
    def calc_cost_estimate(self):
        #todo: check if there has been any changes that require recalculations
        self.cost_estimate =self.results_uncertainty.assign(**{"Level " + str(self.level): self.__class__.__name__})
        return self.cost_estimate

    @param.depends('calc_cost_estimate', watch=True)
    def calc_schedule_estimate(self):
        self.schedule_estimate = self.cost_estimate.agg(start_year = ('FY','min'), end_year=('FY', 'max'))
    
    def _prepare_sim(self):
        if self.u_inputs is not None: 
            self.param.update(**self.u_inputs)
        
    def _end_sim(self):
        if self.u_inputs is not None:
            for key,val in self.u_inputs.items():
                self.param.update(**{key: self.param[key].default})
        self.calc_cost_estimate()
    
    def run_simulation(self, trials=100,clear_previous_sim=True, agg_results = True, agg_columns=[ 'FY']):
        self._prepare_sim()
        if clear_previous_sim: self.sim_results =pd.DataFrame()
        for i in range(trials):
            self._prepare_sim()
            self.calc_cost()
            if agg_results:
                self.sim_results = pd.concat([self.sim_results,
                                              self.cost_estimate.groupby(by=agg_columns )['value_cp'].sum().reset_index().assign(Trial=i)]
                ,ignore_index=True)
            else:
                self.sim_results = pd.concat([self.sim_results,
                                              self.results.assign(Trial=i)
                                              ],
                                              ignore_index=True)
        self._end_sim()
    def run_simulation_parallel(self, trials=100, agg_results=True, agg_columns=['APPN', 'FY']):
        import multiprocessing
        with multiprocessing.Pool() as pool:
            pool.map(self.run_simulation, range(len(self.models)))    
class ParentModel(Model):
    
    
    models = param.List()
    
    def calc_cost(self, run_parallel=False):
        results = []
        if run_parallel:
            import multiprocessing
            with multiprocessing.Pool() as pool:
                pool.map(self.calc_cost_model, range(len(self.models)))
        else:
            
            for model in self.models:
                model.calc_cost()
                results.append(model.results.assign(ModelType = type(model).__name__))
        
            if len(results) >0: self.results = pd.concat(results, ignore_index=True)
        
    def calc_cost_model(self, i):
        self.models[i].calc_cost()
    
    def _add_model(self, model):
        #for new_param in model.param
            #self.
            #self.param.watch(self.calc_cost, ['a'], queued=True, precedence=2)
        self.models.append(model)
        self.calc_cost()
    
    def _prepare_sim(self):
        if self.u_inputs is not None:
            self.param.update(**self.u_inputs)
        for model in self.models:
            model._prepare_sim()
        
    def _end_sim(self):
        if self.u_inputs is not None:
            for key,val in self.u_inputs.items():
                self.param.update(**{key: self.param[key].default})
        
        for model in self.models:
            model._end_sim()
        self.calc_cost()


class ModelApp(param.Parameterized):
    model = param.ClassSelector(Model)

    
    @param.depends('model.cost_estimate')
    def view_summary(self):
        data = self.model.cost_estimate.pivot_table(columns = "FY", values = 'value_cp', aggfunc='sum')
        plt = data.plot(kind='bar')
        return pn.Column(data,pn.Card(plt, title="Plot", sizing_mode='stretch_width'))
    
    @param.depends('model.param')
    def view_outputs(self):
        return pn.widgets.Tabulator(self.model.cost_estimate, header_filters=True)

    def view_model(self):
        
        inputs = [] 
        for p in self.model.param:
            
            if getattr(getattr(self.model, p, None), "__panel__", None):
                print(p, 'has panel')
                inputs.append(
                    pn.Card(
                        getattr(getattr(self.model, p), "__panel__"),
                        title=p, collapsed=True)
                        ) #
            else:
                print(p, 'does not have panel')
                inputs.append(self.model.param[p])
        return pn.Column(
            
            pn.Card(*inputs, title="Inputs"),
            pn.Card(self.view_outputs, title = "Outputs", sizing_mode='stretch_width'),
            sizing_mode='stretch_width'
        )
    
    def view_documentation(self):
        import inspect
        []
        return self.model.__doc__
    def view_graph(self):
        import networkx as nx
        g = Reactive.build_dtree(self.model)
        self.__G__=g
        fig = Reactive.ShowTree(g, lib="matplotlib")
        df=pd.DataFrame.from_records([{"from":e[0], "to":e[1] } for e in g.edges])

        return pn.Row(pn.pane.Matplotlib(fig), df) 
    def __find_nested_params(self):
        p_list = []
        for p in self.model.param:
            if isinstance(self.model.param[p], param.Parameterized):
                print("nested param:", p)
                p_list.append(p)
        return p_list

    def __panel__(self):
        summary = pn.layout.FloatPanel(self.view_summary,
                                       sizing_mode='stretch_both',
                                       position='center-top',
                                       offsety=40,
                                       offsetx=20,
                                    contained=False,
                                    name='Summary: ' + self.model.name,
                                    config = {"headerControls": {"maximize": "remove","close": "remove"}})
        
        return pn.Column( 
            summary,
            pn.Tabs(('Model', self.view_model), ("CEMM", "CEMM") ,("Documentation", self.view_documentation), ("Graph", self.view_graph), sizing_mode='stretch_width'),
            sizing_mode='stretch_width'
        )



# %%
class Inventory(param.Parameterized): #pn.viewable.Viewer
    profile=param.DataFrame(default = pd.DataFrame({'FY':range(2030, 2040), 'quantity' : [5]*5 + [10]*5}))
    delivery_cycle=param.Integer(2)
    service_life=param.Integer(20)

    @property
    def procurement(self):
        return self.profile

    @property
    def delivery(self):
        return self.procurement.FY + self.delivery_cycle

    @property
    def inventory(self):
        return self.delivery
            
    
    def __panel__(self):
        return pn.Card(self.param.profile, title = "Inventory",sizing_mode="stretch_width")        
# %%
class Development(Model):
    cost =param.Number(100)
    duration = param.Number(5)
    start_year=param.Number(2020)
    phased_estimate=pd.DataFrame()


    def __call__(self):
        return self.phased_estimate
    
    @property
    def end_year(self):
        return self.start_year+self.duration

    @param.depends('start_year','duration', 'cost', watch=True, on_init=True)
    def calc_cost(self):
        df = pd.DataFrame(dict(
            #Model = "Development",
            FY = range(int(self.start_year), int(self.end_year)),
            value_cp = [self.cost / self.duration] * (int(self.end_year)- int(self.start_year))
                              )
                         )
        self.results=df
        return self.results
    

class Production(Model):
    '''
    # Learning Curve Analysis:
    This analysis 
    '''
    #def __init__(self, T1,LC, RC, Priors = 0, Inventory=Inventory()):
    T1=param.Number(default=100)
    LC = param.Number( default=.95, bounds=(.6, 1))
    RC = param.Number( default=.95, bounds=(.6, 1))
    Priors = param.Number( default=0)
    Inventory = param.ClassSelector(Inventory, default = Inventory())
    

    def __call__(self):
        return self.lot_cost

    @property
    def Quantities(self):
        print(self.Inventory.procurement.head())
        return self.Inventory.procurement

    @param.depends('T1', 'LC', 'RC', 'Priors', 'Inventory.profile', watch=False)
    def calc_lot_cost(self):
        print("T1:", self.T1)
        df = (self
              .Inventory.profile.assign(T1= self.T1, LC=self.LC, RC= self.RC, Priors = self.Priors)
             )
        df  = (df
              .assign(
                  First = lambda x: x.quantity.cumsum().shift(1).fillna(1) + self.Priors, 
                  Last = lambda x: x.quantity.cumsum()+self.Priors, 
                  midpoint = lambda x: ((x.First + x.Last) * (x.First*x.Last))**.5 / 4,
                  auc = lambda x: x.T1 * x.midpoint**(np.log(x.LC)/2) * x.quantity **(np.log(x.RC)/2), 
                  value_cp = lambda x: x.auc * x.quantity
              )
             )
        self.results = df
        return self.results
    
    @property
    def lot_cost(self):
        return self.calc_lot_cost()

    @param.depends('calc_lot_cost')
    def plot_lot_cost(self):
        return self.lot_cost.hvplot(x='FY', y='auc')

        
    
        
class Demo_Program(ParentModel):
    def __init__(self, Inventory=Inventory()):
        self.Inventory = Inventory
        self.models = [Development(cost=500, duration=10, start_year=2020), Production(Inventory= self.Inventory)]



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
class EVM(Model):
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


class Factor(Model):
    pass


if __name__ =="__main__":
    pass
    #GlobalInputs = Inputs()
    #CostEstimate = CostModel(meta={'Author':"Kevin Joy"})
    #CostEstimate.predict(inputs = GlobalInputs)
# %%
