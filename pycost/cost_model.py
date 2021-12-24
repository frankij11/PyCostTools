import pandas as pd
import param

class GlobalInputs(param.Parameterized):
    BY = param.Integer(2020)
    


class Model(param.Parameterized):
    meta = param.Dict(
        default = {'Analyst': "N/A",
                  'Estimate': "N/A",
                  }
    )
    g_inputs = param.ClassSelector(GlobalInputs, GlobalInputs(), instantiate=False)
    u_inputs = param.Dict()
    results = param.DataFrame(
        default = pd.DataFrame(columns =['Commodity', 'APPN','Inflation', 'Methodology', 'DataSource','BaseYear', 'FY','Value']),
        doc = '''Results container shared by all estimates. This value should be updated using the generic _calc method''',
        columns = set(['Commodity', 'APPN','Inflation', 'Methodology', 'DataSource','BaseYear', 'FY','Value'])
    )
    sim_results = param.List()
    
    def __init__(self, **params):
        super().__init__(**params)
        # Automatically set results to a calculation given defaults
        self._calc()

    def _calc(self):
        print(self.meta['Estimate'], "Not Implement")
        self.results = self.results
    
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
        if clear_previous_sim: self.sim_results =[]
        for i in range(trials):
            self._prepare_sim()
            self._calc()
            if agg_results:
                self.sim_results.append(self.results.groupby(by=agg_columns )['Value'].sum().reset_index().assign(Trial=i))
            else:
                self.sim_results.append(self.results.assign(Trial=i))
        self._end_sim()
    def run_simulation_parallel(self, trials=100, agg_results=True, agg_columns=['APPN', 'FY']):
        import multiprocessing
        with multiprocessing.Pool() as pool:
            pool.map(self.run_simulation, range(len(self.models)))
    
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
import param
import pandas as pd
import numpy as np
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

class LearningCurve(BaseModel):
    '''
    Structure to develop manufacturing estimate
    '''
    pass

class Factor(BaseModel):
    pass


if __name__ =="__main__":
    pass
    #GlobalInputs = Inputs()
    #CostEstimate = CostModel(meta={'Author':"Kevin Joy"})
    #CostEstimate.predict(inputs = GlobalInputs)
# %%
