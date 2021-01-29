# utils
import timeit

# Data Model
import numpy as np
from numpy.lib.shape_base import _replace_zero_by_x_arrays
import pandas as pd
import param

# Visualize
import holoviews as hv
import hvplot.pandas
import panel as pn
import panel.widgets as pnw

# Analysis
import statsmodels.formula.api as smf
import patsy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV

# Pre Processing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Scoring
from sklearn import metrics


# class Analysis:
#     ''' Master Analysis Class. Implement standard analysis process while also

#     '''
#     def __init__(self, df, formulas=[], by=[], test_split = .2, random_state=42, 
#                 model_types=dict(
#                     lm = LinearRegression(),
#                     rf=RandomForestRegressor(), 
#                     lasso=LassoCV(cv=5), 
#                     ridge=RidgeCV(cv=5), 
#                     enet = ElasticNetCV(cv=5)),
#                 steps=[]):
        
#         # store data
#         self.df = df
        
#         self.random_state=42
#         self.test_split = test_split
#         self.train_set = np.random.uniform(size=len(df)) > test_split
        
        
#         # Build unique models
#         self.model_types = model_types
#         self.by = by
#         if type(formulas) != list:
#             self.formulas = [formulas]
#         else:
#             self.formulas = formulas
        

#         # Calculate total number of models
#         # print to user  the number of models expected to be made
#         if len(by) > 0
#             self.total_models = len(model_types) * len(df.groupby(by)) * len(self.formulas)
#         else:
#             self.total_models = len(model_types) * len(self.formulas)
        
#         print(self.total_models)

#         self.steps = steps # list of tuples (name, pipeline)

#         # Store models in dictionaries
#         self.models = dict() # actual models, based on UID
#         self.model_specs = dict() # specs of model, based on UID


#         # Store Summary Statistics
#         self.results = pd.DataFrame(columns=['UID', 'Model',*by, 'Formula','Time', 'RSQ','MSE','AbsErr', 'CV', 'DF', 'MaxError', 'TrainRSQ'])

#     # update based on
#     def eda(self, formulas):
#         # return panel object with graphs and toggles
#         # graphs, correlations, to target variable
#         app = pn.template.BootstrapTemplate(title="KJ's EDA Tool")
#         y,X = patsy.dmatrices(formula, )
#         target = y
#         feature = pnw X[1]
#         by = self.by
#         layout = pn.Card(
#             pn.Column(pn.Card()
#         )


#         pass

#     def add_model(self, models):
#         '''
#         PARAMS: 
#             models: dict with (name = sklearn model )
#         '''

         
#         uid = f'{str(id)}: {mod}'}
#         self.models[uid] = "implement"
    
#     def make_models(self, model_types=self.models_types, formulas=self.formulas, by=self.by):
#         # fit all models
#         try:
#             id = 0
#         except:
#             id = int(max(self.models.keys()).split("_")[1]) + 1

         
#         for f in formulas:
#             for mod in model_types:
#                 for meta,frame in self.df.groupby(by):
#                     id += 1
#                     uid = f'{str(id)}: {mod}'}
                    
#                     self.models[uid] = Pipeline([('preprocess', self.pipeline),
#                                                             (mod, model_types[mod] )]).fit(X,y)
#                     self.model_specs[uid] = dict(model = mod,formula = f, by=meta)

#                     self.models[uid].fit(X,y)


#     def fit(self, model=None):
#         df_test = df[self.test_train]
            
#         if model == None:
#             for mod in self.models:
#                 f = self.model_specs[mod]['formula']

#                 y,X = patsy.dmatrices(self.model_specs[mod]['formula'])
#                 mod.fit(X,y)
#         else:
#             y,X = patsy.dmatrices(self.model_specs[model])
#             model.fit(X,y)
#     def predict(self, model=None, df=None):
#         results = dict()
#         if model == None:
#             for mod in self.models:
#                 y,X = patsy.dmatrices(formula, df, return_type='dataframe')
                
#                 mod.predict(X,y)
#         else:
#             formula = self.formula
#             y,X = patsy.dmatrices(formula, df, return_type='dataframe')
#             model.predict(X,y)


#         return results
#     def report(self, model=None):
#         print('not implemented yet')
#         pass
#     def summary(self):
#         return self.results
#     def feature_importance(self):
#         '''
#         Routine to determine which features are most important
#         self.features = top X: add this to model sets
#         '''
#         pass

#     def update_pipeline(self):
#         pass
    


class Model:
    def __init__(self,df, formula, model=RandomForestRegressor(), test_split = .2, random_state=42):
        self.df_raw = df
        self.formula = formula
        self.model_org = model
        
       
        # Test Train Split
        self.random_state=random_state
        self.test_split = test_split
        
        
        self.y, self.X = patsy.dmatrices(formula, df, return_type='dataframe')
        if test_split >0 & test_split <1:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_split, random_state=self.random_state)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = (self.X, self.X, self.y, self.y)
        self.target_name = self.y.columns
        self.feature_names = self.X.columns

        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                                ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing'))
                                                    ])

                                            # not needed because of patsy ('one_hot', OneHotEncoder())

        numeric_features = self.X.select_dtypes(include=np.number).columns
        categorical_features = self.X.select_dtypes(include='object').columns
        self.preprocessor = ColumnTransformer(
                    transformers=[('num', numeric_transformer, numeric_features),
                                    ('cat', categorical_transformer, categorical_features)
                                            ])   

        self.model = Pipeline([
            ('preprocess', self.preprocessor),
            ('model', model)])
        
        pass

    def fit(self, X= None, y=None):
        if X is None: 
            X=self.X_train
        if y is None: 
            y=self.y_train
        start_time = timeit.default_timer() 
        self.model.fit(self.X, self.y)
        run_time = timeit.default_timer() - start_time
        return self.model
    
    def predict(self, df=pd.DataFrame(), X=None):
        if df.empty:
            if X is None:
                X=self.X
        else:
            y,X = patsy.dmatrices(self.formula, df, return_type='dataframe')
            # Add in code to make more robust
            # for each var in feature_names if does not exists add to X
            # for each var in X and not in feature_names remove from X
            X = X[self.feature_names]

        

        return self.model.predict(X)

    def results(self):
        y_pred=self.model.predict(X=self.X_test)
        y_act=self.y_test
        y_pred_train=self.model.predict(X=self.X_train)
        y_act_train=self.y_train
        
        results = {
            'Model': [self.model['model']],
            'Formula':[self.formula],
            'RunTime':[self.run_time],
            'RSQ': [metrics.r2_score(y_pred, y_act)],
            'MSE': [metrics.mean_squared_error(y_pred, y_act)],
            'AbsErr': [metrics.mean_absolute_error(y_pred, y_act)], 
            'CV': [metrics.mean_squared_error(y_pred, y_act) / np.mean(y_act) ],
            'DF': [len(self.X_train) - len(self.feature_names) ],
            'MaxError': [metrics.max_error(y_pred,y_act)],
            'TrainRSQ':[metrics.r2_score(y_pred_train, y_act_train)]
        }

        return results
