# utils
from datetime import datetime
import datetime as dt
import os
import timeit
import joblib
from tqdm import tqdm


# Data Model
import numpy as np
from numpy.lib.function_base import select
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
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV


# Pre Processing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Scoring
from sklearn import metrics
from sklearn.model_selection import cross_val_score # Cross validated score

class Model:
    # issues:
        # No robust handling of nas
            # drop columns with too many na's
            # impute strategy --> Median, KNN, someother model regression

    # Model Flow:
        # 1. get analysis columns only ( throw away rest)
        # 2. na_preprocess analysis columns (either impute or drop)
        # 3. calculate X based on patsy formula
        # 4. add new features (Poly, Power, etc.)
        # 5. do feature selection
        # 6. fit chosen model

    def __init__(self,df=None,formula=None,target=None, model=RandomForestRegressor(), test_split = .2, random_state=42,handle_na=True,na_processor=None,preprocessor=None,meta_data=dict(title="My Report", desc="N/A",analyst="N/A"), **kwargs):
        # Get attributes
        #self._meta_data = dict(title = title,desc= desc,analyst = analyst, **kwargs)
        self._meta_data = dict(**meta_data, **kwargs)

        if df is None:
            df =self.open_data()
        if (formula is None): # Start Feature selection routine
            if (target is None):
                print(df.columns.tolist())
                target=input("Choose Target Column: ") # df.columns[0]
            
            # Start Feature selection routine
            # Implement pipeline to Add Features / Remove Features
            formula = f"Q('{target}') ~ "
            for var in df.drop(target,axis=1).columns:
                if df[var].dtype is np.number:
                    # Use Q just to be safe
                    formula += f" + Q('{var}')"
                else:
                    var_ratio = len(df[var].unique()) / len(df[var])
                    if var_ratio < .05:
                        formula += f" + C(Q('{var}'))"
        else:
            pass
            #target = patsy.dmatrices(formula, df.loc[0:2])[0].design_info.column
        self.analysis_cols = self.get_formula_cols(formula, df)
        self.target_cols = self.get_formula_cols(formula, df, target_val=True)
        self.feature_cols = self.get_formula_cols(formula, df, feature_vals=True)

        self.df = df[self.analysis_cols]

        self.na_processor = na_processor
        if na_processor is None:
            # NEED TO NOT PROCESS TARGET VARIABLE!!!!!
            num_cols = self.df[self.feature_cols].select_dtypes(include =np.number).columns.tolist()
            obj_cols = self.df[self.feature_cols].select_dtypes(include='object').columns.tolist()
            numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
            categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing'))
                                                    ])
            self.na_processor = ColumnTransformer(
                    transformers=[('num', numeric_transformer, num_cols),
                                    ('cat', categorical_transformer, obj_cols)
                                            ])   

        self.na_processor.fit(self.df)

        if handle_na:
            ##### NOT WORKING
            #print("Fill NA's not implemented yet")
            #print(df.apply(lambda x: sum(x.isna()), axis=1) )
            #self.df = pd.DataFrame(self.na_processor.transform(self.df)) #, columns=self.df.columns)
            pass




        
        self.formula = formula
        self.ModelDate = datetime.now()
       
        # Test Train Split
        self.random_state=random_state
        self.test_split = test_split
        

        self._y, self._X = patsy.dmatrices(formula, df) #, return_type='dataframe')
        self.y = pd.DataFrame(np.asarray( self._y), columns=self._y.design_info.column_names)
        self.X = pd.DataFrame(np.asarray(self._X),columns = self._X.design_info.column_names)
        if (test_split >0.0) & (test_split <1.0):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_split, random_state=self.random_state)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = (self.X, self.X, self.y, self.y)
        self.target_name = self._y.design_info.column_names
        self.feature_names = self._X.design_info.column_names
        self.term_names = self._X.design_info.term_names



        # Preprocessor:
        self.preprocessor = preprocessor
        # All data should be numeric by now
        if preprocessor is None:
            self.preprocessor = Pipeline(
                steps=[
                    ('scaler', StandardScaler() ),
                    ('addPoly', PolynomialFeatures(include_bias=False) ),

                ]
            )
        self.model = Pipeline([
            ('preprocess', self.preprocessor),
            ('model', model)])
        #fit(self)

    def __repr__(self):
        summary = ("Model Summary\n"
                   "  Formula: %s ~ %s\n"
                   "  Model:%s\n"
                   % (self._y.design_info.describe(),
                      self._X.design_info.describe(),
                      self.model))
        
        try:
            results = self.summary()
            print(results)
            
        except:
            summary += "\n\nModel has not been fit to data yet"

        return summary

    def fit(self, X= None, y=None):
        if X is None: 
            X=self.X_train
        if y is None: 
            y=self.y_train.values.ravel()
        start_time = timeit.default_timer() 
        fitted_model=self.model.fit(X,y)
        self.run_time = timeit.default_timer() - start_time
        return self
    
    def predict(self, df=pd.DataFrame(), X=None):
        #(new_x,) = patsy.build_design_matrices([self._X.design_info],
        #                                 df)
        
        if df.empty:
            if X is None:
                X=self.X
        else:
            # Add missing columns as empty            
            add_vars = set(self.df.columns) - set(df.columns)
            for var in add_vars:
                df[var] = np.nan
            X = patsy.build_design_matrices([self._X.design_info],df, return_type='dataframe')[0]
            # Add in code to make more robust
            # for each var in feature_names if does not exists add to X
            # del_vars = set(X) - set(self.feature_names)

            # for each var in X and not in feature_names remove from X
            #X = X[self.feature_names]

        

        return self.model.predict(X)

    def summary(self):
        y_pred=self.model.predict(X=self.X_test)
        y_act=self.y_test
        y_pred_train=self.model.predict(X=self.X_train)
        y_act_train=self.y_train
        
        results = pd.DataFrame({
            'Model': [self.model['model']],
            'Formula':[self.formula],
            'RunTime':[self.run_time],
            'ModelDate': [self.ModelDate],
            'ReportDate': [datetime.now()],
            'RSQ': [metrics.r2_score(y_pred, y_act)],
            'MSE': [metrics.mean_squared_error(y_pred, y_act)],
            'AbsErr': [metrics.mean_absolute_error(y_pred, y_act)], 
            'CV': [metrics.mean_squared_error(y_pred, y_act) / np.mean(y_act)[0] ],
            'DF': [len(self.X_train) - len(self.feature_names) ],
            'MaxError': [metrics.max_error(y_pred,y_act)],
            'TrainRSQ':[metrics.r2_score(y_pred_train, y_act_train)]
        })

        

        return results

    def save(self,name, remove_data=False, compress=3):
        
        if ".joblib" not in name:
            name = name + ".joblib"
        
        obj = self
        if remove_data:
            del obj.X
            del obj.y
            del obj.df_raw
            del obj.X_test
            del obj.X_train
            del obj.y_test
            del obj.y_train
        joblib.dump(obj, name, compress)
        self.save_date = datetime.now()
        print(f"{name} (Model Size): {np.round(os.path.getsize(name) / 1024 / 1024, 2) } MB")
        pass

    def report(self,show=True,**kwargs):
        X = self.X
        app = pn.template.BootstrapTemplate(title="Model Report")
        pn.config.sizing_mode="stretch_width"
        
        
        # Header
        app.header.header_background ='blue'
        #app.header.append(pn.pane.Markdown("# Report"))
        # Side Bar
        
        #inputs = {f"{col}" : pnw.FloatSlider(name=col,start=X[col].min(), end=max(1,X[col].max()), value=X[col].median()) for col in X.columns}
        #for input in inputs:
        #    app.sidebar.append(inputs[input])
        
        
        # Main
        summary_df = self.summary().T
        #summary_df.columns = summary_df.loc[0]
        #summary_df = summary_df.loc[1:]
        preds =pd.DataFrame({"Actual" : self.y.values.ravel(), "Predicted" : self.predict(X=self.X)})
        act_vs_pred = preds.hvplot(x='Predicted', y='Actual',kind='scatter', title='Actual vs Predicted') * hv.Slope(1,0).opts(color='red')
        summary = pn.Row(
                pn.Card(summary_df, title='Summary Statistics', height=500),
                pn.Card(act_vs_pred, title= "Actual Vs Predicted" , height=500)
            )
        pages = pn.Tabs(('Summary', summary), ('Feature Importance', pn.panel("in work")))
        
        app.main.append(pages)
        if show:
            server = app.show("KJ's Model Anlaysis report",threaded=True )
            return (app, server)
        else:
            return app
    def find_knn(self,df, n=5):
        knn = NearestNeighbors(n_neighbors=n).fit(self.X_train)
        X = patsy.build_design_matrices([self._X.design_info],df, return_type='dataframe')[0]
        distances, indices = knn.kneighbors(X)
        X['nn'] = [ind for ind in indices]
        dfs = {}
        dfs2 = []
        for index, row in X.iterrows():
            dfs[index] = {"train": self.X_train.loc[row.nn],
                          "raw_df": self.df.loc[row.nn]
                        }
            
        return dfs

    @staticmethod
    def open_data():
        '''
        Function Returns a dataframe
        '''
        try:
            from tkinter.filedialog import askopenfilename
            filename = askopenfilename()
            if ".csv" in filename.lower():
                df = pd.read_csv(filename)
            elif ".xls" in filename.lower():
                dfs = pd.read_excel(filename,sheet_name=None, engine='openpyxl')
                print("Available Sheets: \n", dfs.keys())
                sheet = input("Which Sheet?") 
                df = dfs[sheet]
                del dfs
            else:
                print("Uknown file type")
                raise()
        except:
            print("could not load data")
            df = pd.DataFrame()
        
        return df
    @staticmethod
    def get_formula_cols(formula, df, target_val=False, feature_vals=False):
        if target_val: formula = formula.split("~")[0]
        if feature_vals: formula = formula.split("~")[1]
        df = df.loc[0:2] # test just the first 2 datapoints so it runs quicker?
        cols = []
        for col in df.columns:
            
            try:
                if target_val | feature_vals:
                    tmp_mod = patsy.dmatrix(formula, df.drop(col, axis=1))
                else:
                    tmp_mod = patsy.dmatrices(formula, df.drop(col, axis=1))
                
            except:
                cols.append(col)
        return cols

class LC_Model(Model):
    def __init__(self, quantity, Lot, grp_cols=[], model=RidgeCV(),First=None, Last=None,unit_no=None,  **kwargs):
        super().__init__(**kwargs)
        # Calculate midpoint
        # create model

        # Add midpoint to analysis data frame

class Models(Model):
    def __init__(self, df,formulas=[], by=[],target=None, models=[LinearRegression(),
                     RandomForestRegressor(), 
                     LassoCV(cv=5), 
                     RidgeCV(cv=5), 
                     ElasticNetCV(cv=5)],test_split=.2, random_state=42,meta_data=dict(title="My Report", desc="N/A",analyst="N/A"),tags={}, **kwargs):
        
        self.meta_data = meta_data
        #super().__init__(**kwargs)
        self.df = df

        if type(formulas) != list:
             self.formulas = [formulas]
        else:
            self.formulas = formulas

        if len(self.formulas)==0:
            if target is None:
                print("No formula or target provided! Please try again")
                
            

        if type(by) != list:
            self.by = [by]
        else:
            self.by = by


        # count number of models necessary to build
        if len(self.by) == 0:
            self.df['GROUP_COLUMN'] = 'all'
            self.by = ['GROUP_COLUMN']
       
       
        self.n_models = len(formulas) * len(models) * len(self.by)
        

        # create master model
        
        # create unique list of models
        _model_specs={}
        _models = {}

        i= 0
        for meta, frame in tqdm( self.df.groupby(by) ):
            for f in self.formulas:
                for mod in models:
                    i+=1
                    tmp_model = Model(df=frame,formula=f, model=mod, **kwargs)
                    _model_specs[f'UID: {i}'] = dict(
                        **self.meta_data,
                        **tags,
                        **{col:meta for col,meta in zip(by, meta)},
                        formula = f, 
                        model_spec = mod.__repr__(), 
                        IsFitted=False,
                        Model = tmp_model
                    )
                    
                    #_models[f'UID: {i}'] = Model(df=frame,fromula=f, model=mod, **kwargs)
        
        
        #self._model_specs = _model_specs
        self.model_summary = pd.DataFrame(_model_specs).T
        #self.models = _models
        

    def add_models(self,model=None, **kwargs):
        # get model summary

        # concat model summary to self
        self.model_summary
        pass
    
    def delete_models(self, model, **kwargs):
        pass

    def fit(self, X=None,y=None, timeout_in_seconds=90):
        start_time = dt.datetime.now() # timeit.default_timer()
        time_limit = start_time + dt.timedelta(0,timeout_in_seconds) 
        i  = 0
        for index,row in self.model_summary[['IsFitted','Model']].iterrows():
            if dt.datetime.now() < time_limit: # timeit.default_timer()
                # Run Model
                if not row.IsFitted:
                    i+=1
                    # Fit model and tell model summary it is fitted
                    fitted_model=row.Model.fit()
                    row.IsFitted = True
                    # delete unnecessary data attributes?

                
            else:
                print("Time Limit Reachs")
                print(f"{i} Models were fit")
                print(f"{self.n_models-i} Models still require fit")
                break
        self.run_time = dt.datetime.now() - start_time # timeit.default_timer()
 
        return 
    
    def predict(self,df, best=True):
        y_all = []
        for meta,frame in df.groupby(self.by):
            # filter model specs for meta
            
            # get uid's for available models

            # if no models available_use master model? or return np.nan?

            # else for each UID run prediction

            pass
            # 
        reuslts = pd.DataFrame()
        return results

if __name__ == "__main__":
    import pycost as ct
    df = ct.jic
    df['Year'] = pd.to_numeric( df['Year'], errors='coerce')
    f = "Raw ~ Version+Service+tags+Indice +Year"
    m = Model(df,f)
    myModels =Models(df, "Raw ~ Year", by=["Version", "Service", "tags", "Indice"], tags={'JIC': 2020})
    myModels.fit(timeout_in_seconds=5)
    myModels.fit(timeout_in_seconds=10)

     

    model_types = dict(
                        lm = LinearRegression(),
                        rf=RandomForestRegressor(), 
                        lasso=LassoCV(cv=5), 
                        ridge=RidgeCV(cv=5), 
                        enet = ElasticNetCV(cv=5))
    mods = []
    results = pd.DataFrame()
    for mod in model_types:    
        m=Model(df=ct.jic, formula=f, model=model_types[mod])
        m.fit()
        #print(m.feature_names)
        #print(m.summary().T)
        mods.append(m)
        results = pd.concat([results, m.summary()], ignore_index=True)

    print(results.T)
    #print(mods[0])

    new_df = df.loc[0:2].assign(Year=range(2090,2093))
    for mod in mods:
        print(mod.model['model'],mod.predict(new_df))
        print("closest points")
        print([item for key, item in mod.find_knn(new_df, 5).items()])
    #app,server = mods[0].report()
