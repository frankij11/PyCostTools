import pandas
import param

class BaseModel(param.paramerterized):
    '''
    Base model for building a cost estimate. Similar to an
    empty worksheet. All other models should extend this class
    '''
    def __init___(self, name, categories, analyst, **kwargs):
        pass

    def fit(self, **kwargs):
        '''
        Implement model
        '''
        return self

    def predict(self, inputs):
        '''
        Implement predict
        '''
        
        return df
    
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
        

class CostModel:
    '''
    Collection of estimates, similar to a workbook in Excel
    '''
    pass

class Inputs:
    pass

class Development(BaseModel):


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

class Factor(BaseModel)