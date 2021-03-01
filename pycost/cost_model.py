import pandas as pd
import param

class BaseModel(param.Parameterized):
    '''
    Base model for building a cost estimate. Similar to an
    empty worksheet. All other models should extend this class
    '''
    meta= param.Dict(
        Analyst = param.String("Uknown"),
        WBS = param.string('Uknown') 
        )

    inputs = Inputs()

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
    global_inputs = param.Dict(
        ProgramName = param.String("NA"),
        BY = param.Integer(2020, bounds=(1970,2060) ),
        EstimateName= param.String("NA")
        )
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
    GlobalInputs = Inputs()
    CostEstimate = CostModel(meta={'Author':"Kevin Joy"})
    CostEstimate.predict(inputs = GlobalInputs)
# %%
