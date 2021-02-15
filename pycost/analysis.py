# my modules
from pycost import clean

# utils
from datetime import datetime
import datetime as dt
import os
import timeit
import joblib
from tqdm import tqdm
import copy

# Data Model
import numpy as np
from numpy.lib.function_base import select
from numpy.lib.shape_base import _replace_zero_by_x_arrays
import pandas as pd
#import param

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
from sklearn.model_selection import RandomizedSearchCV


# Pre Processing
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, PolynomialFeatures, PowerTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Scoring
from sklearn import metrics
from sklearn.model_selection import cross_val_score  # Cross validated score


class TEST:
    def __init__(self) -> None:
        print("this is a test")


class Model:
    # issues:
    # No robust handling of nas
    # drop columns with too many na's
    # impute strategy --> Median, KNN, someother model regression

    '''
    Master modeling class that handles many of the details of Machine Learning.
    Implements the 
  
    Model Flow:
    1. get analysis columns only ( throw away rest)
    2. na_preprocess analysis columns (either impute or drop)
    3. calculate X based on patsy formula
    4. add new features (Poly, Power, etc.)
    5. do feature selection
    6. fit chosen model
    
    Example:
    df = pd.DataFrame({'y': [1,2,3,4,5], 'x1': [2,4,6,8,10], 'x2': ["a", "b","b","a","a"]})
    myModel = Model(df, "y~x1+x2-1", model= LinearRegression(),
        meta_data={
            'title': "Example Analysis",
            'desc': "Do some anlaysis"}
            )
    myModel.fit().summary()
    myModel.predict(new_df)
    myModel.save("myModel")


    # load data
    loadedModel = Model.load("myModel")
    
    '''

    def __init__(self, df=None, formula=None, target=None, model=RandomForestRegressor(), test_split=.2, random_state=42, handle_na=True, na_processor=None, preprocessor=None, meta_data=dict(title="My Report", desc="N/A", analyst="N/A"), **kwargs):
        # Get attributes
        #self._meta_data = dict(title = title,desc= desc,analyst = analyst, **kwargs)
        self._meta_data = dict(**meta_data, **kwargs)

        if df is None:
            df = self.open_data()
        if (formula is None):  # Start Feature selection routine
            if (target is None):
                print(df.columns.tolist())
                target = input("Choose Target Column: ")  # df.columns[0]

            # Start Feature selection routine
            # Implement pipeline to Add Features / Remove Features
            formula = f"Q('{target}') ~ "
            for var in df.drop(target, axis=1).columns:
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
        self.feature_cols = self.get_formula_cols(
            formula, df, feature_vals=True)

        self.df = df[self.analysis_cols]

        self.na_processor = na_processor
        if na_processor is None:
            # NEED TO NOT PROCESS TARGET VARIABLE!!!!!
            # na_processor = ImputeNA()
            num_cols = self.df[self.feature_cols].select_dtypes(
                include=np.number).columns.tolist()
            obj_cols = self.df[self.feature_cols].select_dtypes(
                exclude=np.number).columns.tolist()
            numeric_transformer = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='median'))])
            categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing'))
                                                      ])
            self.na_processor = ColumnTransformer(
                transformers=[('num', numeric_transformer, num_cols),
                              ('cat', categorical_transformer, obj_cols)
                              ])

        self.na_processor.fit(self.df)

        self.handle_na = handle_na
        if self.handle_na:
            # NOT WORKING
            #print("Fill NA's not implemented yet")
            #print(df.apply(lambda x: sum(x.isna()), axis=1) )
            self.num_cols = num_cols
            self.obj_cols = obj_cols
            self.df = pd.concat([
                df[[*self.target_cols]],
                pd.DataFrame(self.na_processor.transform(
                    self.df), columns=[*num_cols, *obj_cols])
            ], axis=1)

        self.formula = formula
        self.ModelDate = datetime.now()

        # Test Train Split
        self.random_state = random_state
        self.test_split = test_split

        # get y, X to fit data on
        # , return_type='dataframe')
        self.y, self.X = patsy.dmatrices(formula, self.df)
        self.column_names = self.X.design_info.column_names

        if (test_split > 0.0) & (test_split < 1.0):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=self.test_split, random_state=self.random_state)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = (
                self.X, self.X, self.y, self.y)

        # Preprocessor:
        self.preprocessor = preprocessor
        # All data should be numeric by now
        if preprocessor is None:
            self.preprocessor = Pipeline(
                steps=[
                    ('dateCleaner', clean.DateTransform()),
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    #('addPoly', PolynomialFeatures(include_bias=False) )

                ]
            )
        self.model = Pipeline([
            ('preprocess', self.preprocessor),
            ('model', model)])
        # fit(self)

    def __repr__(self):
        summary = ("Model Summary\n"
                   f"  Formula: {self.formula}\n"
                   f"  Model:{self.model.__str__()}\n"
                   )

        try:
            results = self.summary()
            # print(results)

        except:
            summary += "\n\nModel has not been fit to data yet"

        return summary

    def fit(self):
        X = self.X_train
        y = np.asarray(self.y_train).ravel()
        start_time = timeit.default_timer()
        self.model = self.model.fit(X, y)
        self.run_time = timeit.default_timer() - start_time
        return self

    def predict(self, df=pd.DataFrame(), X=None):
        # (new_x,) = patsy.build_design_matrices([self._X.design_info],
        #                                 df)

        if df.empty:
            if X is None:
                X = self.X
        else:
            df = df.copy()

            # need away to impute missing columns
            # df = self.na_processor.transform(df)

            #X = patsy.dmatrices(self.formula, df, return_type='dataframe')[1]
            if self.handle_na:
                df = pd.DataFrame(self.na_processor.transform(
                    df), columns=[*self.num_cols, *self.obj_cols])
            X = patsy.build_design_matrices([self.X.design_info], df)

            # Add in code to make more robust
            #add_vars = set(self.X.columns) - set(X.columns)
            #X[add_vars] = np.nan

            #del_vars = set(X) - set(self.X.columns.tolist())
            # for each var in X and not in feature_names remove from X

            #X.drop(del_vars, inplace=True)

            #X = X[self.feature_names]

        return self.model.predict(X)

    @staticmethod
    def stats(model, X_test, y_test, X_train=None, y_train=None):
        y_pred = model.predict(X=X_test)
        results = pd.DataFrame({
            'Model': [model.__str__()],
            'Formula': [np.nan],
            'RunTime': [np.nan],
            'ModelDate': [np.nan],
            'ReportDate': [datetime.now()],
            'RSQ': [metrics.r2_score(y_pred, y_test)],
            'MSE': [metrics.mean_squared_error(y_pred, y_test)],
            'AbsErr': [metrics.mean_absolute_error(y_pred, y_test)],
            'CV': [metrics.mean_squared_error(y_pred, y_test) / np.mean(y_test)],
            'MaxError': [metrics.max_error(y_pred, y_test)]
        })

        if (not X_train is None) & (not y_train is None):
            y_pred_train = model.predict(X_train)
            Train_Info = pd.DataFrame({
                'DF': [X_train.shape[0] - X_train.shape[1]],
                'TrainRSQ': [metrics.r2_score(y_pred_train, y_train)],
                'TrainY_Mean': np.mean(y_train),
                'TrainY_STD': np.std(y_train)

            })
        else:
            Train_Info = pd.DataFrame()

        results = pd.concat([results, Train_Info], axis=1)

        return results

    def summary(self):

        s = self.stats(
            model=self.model,
            X_test=self.X_test,
            y_test=self.y_test,
            X_train=self.X_train,
            y_train=self.y_train
        )

        results = s.assign(
            Model=self.model['model'],
            Formula=self.formula,
            RunTime=self.run_time,
            ModelDate=self.ModelDate
        )

        y_test = self.y_test
        y_pred = self.model.predict(self.X_test)
        y_pred_train = self.model.predict(self.X_train)

        results = pd.DataFrame({
            'Model': [self.model['model']],
            'Formula': [self.formula],
            'RunTime': [self.run_time],
            'ModelDate': [self.ModelDate],
            'ReportDate': [datetime.now()],
            'RSQ': [metrics.r2_score(y_pred, y_test)],
            'MSE': [metrics.mean_squared_error(y_pred, y_test)],
            'AbsErr': [metrics.mean_absolute_error(y_pred, y_test)],
            'CV': [metrics.mean_squared_error(y_pred, y_test) / np.mean(y_test)],
            'DF': [self.X_train.shape[0] - self.X_train.shape[1]],
            'MaxError': [metrics.max_error(y_pred, y_test)],
            'TrainRSQ': [metrics.r2_score(y_pred_train, self.y_train)]
        })

        return results

    def save(self, name, remove_data=False, compress=3):

        if ".joblib" not in name:
            name = name + ".joblib"

        obj = copy.deepcopy(self)
        del obj.y, obj.X
        if remove_data:
            #del obj.X
            #del obj.y
            #del obj.df
            del obj.X_test
            del obj.X_train
            del obj.y_test
            del obj.y_train
        obj.save_date = datetime.now()
        joblib.dump(obj, name, compress)
        self.save_date = datetime.now()
        print(
            f"{name} (Model Size): {np.round(os.path.getsize(name) / 1024 / 1024, 2) } MB")
        pass

    @staticmethod
    def load(file_name):
        try:
            obj = joblib.load(file_name)
        except:
            try:
                if ".joblib" not in file_name:
                    file_name = file_name + ".joblib"
                    obj = joblib.load(file_name)
            except:
                raise("Could not find file")
                pass
            raise("Could not find file")
            pass

        try:
            # , return_type='dataframe')
            obj.y, obj.X = patsy.dmatrices(obj.formula, obj.df)
        except:
            raise("Could not create formula object. is df missing? Try adding data")
        return obj

    def report(self, show=True, **kwargs):
        X = self.X
        app = pn.template.BootstrapTemplate(title="Model Report")
        pn.config.sizing_mode = "stretch_width"

        # Header
        app.header.header_background = 'blue'
        # app.header.append(pn.pane.Markdown("# Report"))
        # Side Bar

        #inputs = {f"{col}" : pnw.FloatSlider(name=col,start=X[col].min(), end=max(1,X[col].max()), value=X[col].median()) for col in X.columns}
        # for input in inputs:
        #    app.sidebar.append(inputs[input])

        # Main
        summary_df = self.summary().T
        #summary_df.columns = summary_df.loc[0]
        #summary_df = summary_df.loc[1:]
        preds = pd.DataFrame(
            {"Actual": self.y.values.ravel(), "Predicted": self.predict(X=self.X)})
        act_vs_pred = preds.hvplot(x='Predicted', y='Actual', kind='scatter',
                                   title='Actual vs Predicted') * hv.Slope(1, 0).opts(color='red')
        summary = pn.Row(
            pn.Card(summary_df, title='Summary Statistics', height=500),
            pn.Card(act_vs_pred, title="Actual Vs Predicted", height=500)
        )
        pages = pn.Tabs(('Summary', summary),
                        ('Feature Importance', pn.panel("in work")))

        app.main.append(pages)
        if show:
            server = app.show("KJ's Model Anlaysis report", threaded=True)
            return (app, server)
        else:
            return app

    def find_knn(self, df, n=5):
        knn = NearestNeighbors(n_neighbors=n).fit(self.X_train)
        X = patsy.build_design_matrices(
            [self._X.design_info], df, return_type='dataframe')[0]
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
                dfs = pd.read_excel(
                    filename, sheet_name=None, engine='openpyxl')
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
        if target_val:
            formula = formula.split("~")[0]
        if feature_vals:
            formula = formula.split("~")[1]
        # test just the first 2 datapoints so it runs quicker?
        df = df.loc[0:2]
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
    def __init__(self, quantity, Lot, grp_cols=[], model=RidgeCV(), First=None, Last=None, unit_no=None,  **kwargs):
        super().__init__(**kwargs)
        # Calculate midpoint
        # create model

        # Add midpoint to analysis data frame


class Models:
    def __init__(self, df, formulas=[], by=[], target=None, models=[LinearRegression(),
                                                                    RandomForestRegressor(),
                                                                    LassoCV(
                                                                        cv=5),
                                                                    RidgeCV(
                                                                        cv=5),
                                                                    ElasticNetCV(cv=5)], test_split=.2, random_state=42, meta_data=dict(title="My Report", desc="N/A", analyst="N/A"), tags={}, **kwargs):

        self.ModelDate = datetime.now()
        self.Models = {}
        self.db = pd.DataFrame()
        self.data = [df]
        self.results = pd.DataFrame()
        self.n_models = 0
        self.run_time = dt.datetime.now() - dt.datetime.now()

        # get all args that were passed
        args = locals()
        del args['self']
        del args['kwargs']

        # Build Models and Add Them to DB
        self.build_models(**args, **kwargs)

    def add_models(self, List_of_Models=[]):
        if type(List_of_Models) != list:
            List_of_Models = [List_of_Models]
        for model in List_of_Models:
            if type(model) == type(self):
                for mod in model.Models:
                    new_id = max(self.Models+1)

                    # Add to Model Collections
                    self.Models[new_id] = model.Models[mod]

                    # Add to Model DB
                    row = model.db[mod].copy()
                    row.index = new_id
                    self.db = pd.concat(
                        [self.db[new_id], row], axis=0, sort=False, join='outer')

    def build_models(self, df, formulas=[], by=[], target=None, models=[LinearRegression(),
                                                                        RandomForestRegressor(),
                                                                        LassoCV(
                                                                            cv=5),
                                                                        RidgeCV(
                                                                            cv=5),
                                                                        ElasticNetCV(cv=5)], test_split=.2, random_state=42, meta_data=dict(title="My Report", desc="N/A", analyst="N/A"), tags={}, **kwargs):

        df = df.copy()
        if type(formulas) != list:
            formulas = [formulas]
        else:
            formulas = formulas

        if len(formulas) == 0:
            if target is None:
                print("No formula or target provided! Please try again")

        if type(by) != list:
            by = [by]
        else:
            by = by

        # count number of models necessary to build
        if len(by) == 0:
            df['GROUP_COLUMN'] = 'all'
            by = ['GROUP_COLUMN']

        n = len(formulas) * len(models) * len(df.groupby(by))

        print(f"{n} Models are being prepared to be built")
        # Maybe try to build one random model and timeit to forecast
        if n > 200:
            cont = input("Do you want to continue?\n Y/N")
            if cont.lower() not in ["yes", "y"]:
                pass
        # create master model
        # create unique list of models
        _model_specs = {}
        _models = {}

        try:
            i = max(self.Models)+1
        except:
            i = 0
        for meta, frame in tqdm(df.groupby(by)):
            for f in formulas:
                for mod in models:
                    i += 1
                    # aggegrate frame
                    # Example: frame.agg

                    if by == ["GROUP_COLUMN"]:
                        tmp_by = []
                    else:
                        tmp_by = by
                    tmp_model = Model(df=frame, formula=f, model=mod, **
                                      kwargs, **{col: meta for col, meta in zip(by, meta)})

                    self.Models[i] = Model(
                        df=frame, formula=f, model=mod, **kwargs, **{col: meta for col, meta in zip(by, meta)})
                    _model_specs[i] = dict(
                        **meta_data,
                        **tags,
                        **{col: meta for col, meta in zip(by, meta)},
                        Formula=f,
                        ModelType=mod.__repr__(),
                        Model=Model(df=frame, formula=f, model=mod, **kwargs,
                                    **{col: meta for col, meta in zip(by, meta)}),
                        Target=self.Models[i].target_cols,
                        Features=self.Models[i].feature_cols,
                        AnalysisColumns=self.Models[i].analysis_cols,
                        BY=tmp_by
                        # IsFitted=False


                    )

                    #_models[f'UID: {i}'] = Model(df=frame,fromula=f, model=mod, **kwargs)

        #self._model_specs = _model_specs
        self.db = pd.concat([self.db, pd.DataFrame(_model_specs).T])
        #self.models = _models
        pass

    def delete_models(self, model, **kwargs):
        pass

    def fit(self, X=None, y=None, timeout_in_seconds=90, verbose=1):
        start_time = dt.datetime.now()  # timeit.default_timer()
        time_limit = start_time + dt.timedelta(0, timeout_in_seconds)
        i = 0
        for index, row in self.db.iterrows():
            mod = row.Model
            if dt.datetime.now() < time_limit:  # timeit.default_timer()
                # Run Model
                try:
                    p = mod.predict()
                    IsFitted = True
                except:
                    IsFitted = False
                if not IsFitted:
                    i += 1
                    # Fit model and tell model summary it is fitted
                    #self.db.iloc[mod.index,"Model" ]
                    self.db.at[index, "Model"] = self.db.at[index,
                                                            "Model"].fit()
                    # self.Models[mod]=self.Models[mod].fit()
                    # delete unnecessary data attributes?

            else:
                print("Time Limit Reached")
                print(f"{i} Models were fit")
                print(f"{len(self.Models)-i} Models still require fit")
                return self
        self.run_time = self.run_time + \
            (dt.datetime.now() - start_time)  # timeit.default_timer()
        print(
            f"{i} Models were fit \nAll models have been fitted and ready for predictions")
        return self

    def predict(self, df=None, by=[], best=True):
        y_all = pd.DataFrame()

        if df is None:
            df = self.df.copy()
        # analysis_col and by cols == model analysis col
        analysis_cols = []

        for meta, frame in df.groupby(self.by):
            # filter model specs for meta
            q = []
            for i in range(len(meta)):
                q.append(f"(`{self.by[i]}` == '{meta[i]}')")
                #q.append( )
            q_str = " & ".join(q)

            tmp_summary = self.model_summary.query(f"{q_str}")
            # get uid's for available models
            print(f"{len(tmp_summary)} Models found")
            # if no models available_use master model? or return np.nan?
            tmp_index = tmp_summary.index.tolist()

            # get prediction for all models
            tmp_preds = pd.DataFrame(index=frame.index)
            i = 0
            for mod in tmp_index:
                i += 1
                try:
                    # frame
                    tmp_preds[f"Prediction_{i}"] = self.model_summary.Model[mod].predict(
                    )

                except:
                    tmp_preds[f"Prediction_{i}"] = np.nan

            # else for each UID run prediction
            y_all = pd.concat([y_all, tmp_preds])
            # pass
            #
        #results = pd.DataFrame(dict(Predictions=y_all))
        return y_all

    def report(self, show=True, **kwargs):
        #X = self.X
        app = pn.template.BootstrapTemplate(title="Model Report")
        pn.config.sizing_mode = "stretch_width"

        # Header
        app.header.header_background = 'blue'

        # Side Bar
        inputs = dict()
        for col in self.db.Features[0]:
            if self.df[col].dtype == 'object':
                inputs[col] = pn.widgets.Select(
                    name=col, value=self.df[col][0], options=self.df[col].unique().tolist())
            else:
                inputs[col] = pn.widgets.FloatSlider(
                    name=col, value=self.df[col].median())

        widgets = pn.WidgetBox("# Model Inputs", *[inputs[w] for w in inputs])
        app.sidebar.append(widgets)
        # Main
        summary_df = self.db
        #summary_df.columns = summary_df.loc[0]
        #summary_df = summary_df.loc[1:]
        #preds =pd.DataFrame({"Actual" : self.df[self.Target[0]], "Predicted" : self.predict()})
        #act_vs_pred = preds.hvplot(x='Predicted', y='Actual',kind='scatter', title='Actual vs Predicted') * hv.Slope(1,0).opts(color='red')
        summary = pn.Row(
            pn.Card(summary_df, title='Summary Statistics', height=500)
            #pn.Card(act_vs_pred, title= "Actual Vs Predicted" , height=500)
        )
        pages = pn.Tabs(('Summary', summary),
                        ('Feature Importance', pn.panel("in work")))

        app.main.append(pages)
        if show:
            server = app.show("KJ's Model Anlaysis report", threaded=True)
            return (app, server)
        else:
            return app


class AutoPipeline:
    def __init__(
            self, df=None, formula=None, target=None, test_split=.2, random_state=42,
            handle_na=True, na_processor=None, preprocessor=None,
            meta_data=dict(title="My Report", desc="N/A", analyst="N/A"),
            scoring_function='neg_mean_squared_error', n_iter=50,
            **kwargs) -> None:

        args = locals()
        del args['self']
        del args['kwargs']
        for arg in args:
            self.__setattr__(arg, args[arg])

    @staticmethod
    def build_pipeline(X_train):
        categorical_values = []

        cat_subset = X_train.select_dtypes(
            include=['object', 'category', 'bool'])

        for i in range(cat_subset.shape[1]):
            categorical_values.append(
                list(cat_subset.iloc[:, i].dropna().unique()))

        date_pipeline = Pipeline([
            ('dateFeatures', clean.DateTransform())
        ])

        num_pipeline = Pipeline([
            ('cleaner', SimpleImputer()),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('cleaner', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse=False, categories=categorical_values))
        ])

        preprocessor = ColumnTransformer([
            ('numerical', num_pipeline, make_column_selector(
                dtype_exclude=['object', 'category', 'bool'])),
            ('categorical', cat_pipeline, make_column_selector(
                dtype_include=['object', 'category', 'bool']))
        ])

        return preprocessor


class AutoRegressionTrees:
    def __init__(self, scoring_function='neg_mean_squared_error', n_iter=50) -> None:
        self.scoring_function = scoring_function
        self.n_iter = n_iter

    def fit(self, X, y):
        X_train = X
        y_train = y

        categorical_values = []

        cat_subset = X_train.select_dtypes(
            include=['object', 'category', 'bool'])

        for i in range(cat_subset.shape[1]):
            categorical_values.append(
                list(cat_subset.iloc[:, i].dropna().unique()))

        date_pipeline = Pipeline([
            ('dateFeatures', clean.DateTransform())
        ])

        num_pipeline = Pipeline([
            ('cleaner', SimpleImputer()),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('cleaner', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse=False, categories=categorical_values))
        ])

        preprocessor = ColumnTransformer([
            ('numerical', num_pipeline, make_column_selector(
                dtype_exclude=['object', 'category', 'bool'])),
            ('categorical', cat_pipeline, make_column_selector(
                dtype_include=['object', 'category', 'bool']))
        ])

        model_pipeline_steps = []
        # model_pipeline_steps.append(('dateFeatures',date_pipeline))
        model_pipeline_steps.append(('preprocessor', preprocessor))
        model_pipeline_steps.append(
            ('feature_selector', SelectKBest(f_regression, k='all')))
        model_pipeline_steps.append(('estimator', RandomForestRegressor()))
        model_pipeline = Pipeline(model_pipeline_steps)

        total_features = preprocessor.fit_transform(X_train).shape[1]

        optimization_grid = []

        # Random Forest
        optimization_grid.append({
            'preprocessor__numerical__scaler': [None],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, round(total_features/10))) + ['all'],
            'estimator': [RandomForestRegressor(random_state=0)],
            'estimator__n_estimators': np.arange(5, 500, 10),
            'estimator__criterion': ['mse', 'mae']
        })

        # Gradient boosting
        # optimization_grid.append({
        #     'preprocessor__numerical__scaler':[None],
        #     'preprocessor__numerical__cleaner__strategy':['mean','median'],
        #     'feature_selector__k': list(np.arange(1,total_features,5)) + ['all'],
        #     'estimator':[GradientBoostingClassifier(random_state=0)],
        #     'estimator__n_estimators':np.arange(5,500,10),
        #     'estimator__learning_rate':np.linspace(0.1,0.9,20),
        # })

        search = RandomizedSearchCV(
            model_pipeline,
            optimization_grid,
            n_iter=self.n_iter,
            scoring=self.scoring_function,
            n_jobs=-1,
            random_state=0,
            verbose=3,
            cv=5
        )

        search.fit(X_train, y_train)
        self.best_estimator_ = search.best_estimator_
        self.best_pipeline = search.best_params_

    def predict(self, X, y=None):
        return self.best_estimator_.predict(X)

    def save(self, fname, compress=3):

        pass

    def summary(self):
        return Model.stats(self.X_test, self.y_test, self.X_train, self.y_train)


class AutoRegressionLinear:
    def __init__(self, scoring_function='neg_mean_squared_error', n_iter=50):
        self.scoring_function = scoring_function
        self.n_iter = n_iter

        # Impute
        # Drop Columns
        # scale
        # Add Features
        # Feature Selection
        # Estimators
        # Grid Search, Random Search

    def fit(self, X, y):
        X_train = X
        y_train = y

        categorical_values = []

        cat_subset = X_train.select_dtypes(
            include=['object', 'category', 'bool'])

        for i in range(cat_subset.shape[1]):
            categorical_values.append(
                list(cat_subset.iloc[:, i].dropna().unique()))

        date_pipeline = Pipeline([
            ('dateFeatures', clean.DateTransform())
        ])

        num_pipeline = Pipeline([
            ('cleaner', SimpleImputer()),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('cleaner', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse=False, categories=categorical_values))
        ])

        preprocessor = ColumnTransformer([
            ('numerical', num_pipeline, make_column_selector(
                dtype_exclude=['object', 'category', 'bool'])),
            ('categorical', cat_pipeline, make_column_selector(
                dtype_include=['object', 'category', 'bool']))
        ])

        model_pipeline_steps = []
        # model_pipeline_steps.append(('dateFeatures',date_pipeline))
        model_pipeline_steps.append(('preprocessor', preprocessor))
        model_pipeline_steps.append(
            ('feature_selector', SelectKBest(f_regression, k='all')))
        model_pipeline_steps.append(('estimator', LinearRegression()))
        model_pipeline = Pipeline(model_pipeline_steps)

        total_features = preprocessor.fit_transform(X_train).shape[1]

        optimization_grid = []

        # Linear Regression
        optimization_grid.append({
            'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, round(total_features/10))) + ['all'],
            'estimator': [LinearRegression()]
        })

        # Regularized Regression
        optimization_grid.append({
            'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, round(total_features/10))) + ['all'],
            'estimator': [ElasticNetCV()],
            'estimator__l1_ratio': [0.01, .1, .5, .7, .9, .95, .99, 1],
            'estimator__n_alphas': [100]
        })
        search = RandomizedSearchCV(
            model_pipeline,
            optimization_grid,
            n_iter=self.n_iter,
            scoring=self.scoring_function,
            n_jobs=-1,
            random_state=0,
            verbose=3,
            cv=5
        )

        search.fit(X_train, y_train)
        self.best_estimator_ = search.best_estimator_
        self.best_pipeline = search.best_params_

    def predict(self, X, y=None):
        return self.best_estimator_.predict(X)


if __name__ == "__main__":
    import pycost as ct
    df = ct.jic
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    X = df.drop(["Raw", "Weighted"], axis=1)
    y = df.Raw
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    f = "Raw ~ Version+Service+tags+Indice +Year"
    #autoM = AutoRegressionLinear(n_iter=50)
    # autoM.fit(X=X_train,y=y_train)
    # print(Model.stats(autoM,X_train,y_train,X_test,y_test).T)
    m = Model(df, f).fit()
    myModels = Models(df, "Raw ~ Year + tags+ Indice",
                      by=["Version", "Service"], handle_na=False, tags={'JIC': 2020})
    myModels.fit(timeout_in_seconds=5)
    # myModels.fit(timeout_in_seconds=10)
    print(myModels.predict(df).head())
