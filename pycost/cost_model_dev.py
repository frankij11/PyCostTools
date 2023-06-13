import pandas as pd

class GlobalInputs:
    def __init__(self, BY=2023,dol_units=1_000):
        self.BY = BY
        self.dol_units = dol_units

class BaseModel():
    
    @property
    def name(self):
        if hasattr(self, "_name"):
            return self._name
        else:
            self._name = self.__class__.__name__ + '_' + str(id(self))
            return self._name
    
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
        
    @property
    def estimate(self):
        #check if there has been any changes that require recalculations
               
        self._estimate =self().assign(**{"Level " + str(self.level): self.name})
        return self._estimate
    
    def make_report(self):
        import panel as pn
        pn.extension(sizing_mode='stretch_width', align='center')
        col_index = pn.widgets.MultiChoice(value=['Level 1','class'],options=self.estimate.columns.tolist(),name='Group By:')
        col_cols = pn.widgets.Select(value='FY',options=self.estimate.columns.tolist(),name='Columns')
        
        @pn.depends(col_index=col_index,col_cols=col_cols)
        def est_summary(col_index, col_cols):
            return pn.Column(
                pn.Card(pn.widgets.Tabulator(self.estimate.pivot_table(index=col_index,columns=col_cols, values='value',aggfunc='sum' )), align='center'),
                pn.Card(pn.pane.Plotly(self.estimate.plot(x=col_cols,color=col_index[0], y='value',kind='bar' )))
            )
        app = pn.template.BootstrapTemplate(title=self.__class__.__name__)
        app.main.append(pn.Column(pn.Row(col_index,col_cols),est_summary))
        #self.app=app
        return app        
class ParentModel(BaseModel):
    
    
    
    def __call__(self):
        df = pd.DataFrame()
        for model in self.sub_models:
            model.level = self.level + 1
            df = df.append(model.estimate, ignore_index=True)
        return df
        
        
    def add_models(self, models):
        if isinstance(list):
            self.sub_models = self.sub_models + models
    
    @property
    def sub_models(self):
        if hasattr(self,"_sub_models"):
            return self._sub_models
        else:
            self._sub_models = []
            return self._sub_models
    @sub_models.setter
    def sub_models(self,value):
        self._sub_models = value
    
    #def make_report(self):
    #    pass
        
        
    
        
