# my modules
from pycost import learn

# utils
from datetime import datetime
import datetime as dt
import os
import timeit
import joblib
from tqdm import tqdm
import copy
import logging
import sys
import traceback

from pycost.analysis import process
from pycost.cost.utils.logging import setup_logging

# Initialize the logger with default settings
logger = setup_logging(log_level=logging.DEBUG)

# Data Model
import numpy as np
import pandas as pd
#import param

# Visualize
try:
    import holoviews as hv
    import hvplot.pandas
    import panel as pn
    import panel.widgets as pnw
except ImportError:
    logger.warning("Holoviews, hvplot, panel, and panel.widgets are not installed. Some features may not work.")

# Analysis
#import statsmodels.formula.api as smf
import patsy
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import RandomizedSearchCV


# Pre Processing
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, PolynomialFeatures, PowerTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, RegressorMixin
# Scoring
from sklearn import metrics
from sklearn.model_selection import cross_val_score  # Cross validated score

# Utils
from sklearn.utils.validation import check_is_fitted


__all__ = ["Model","LC_Model", "Models","LC_Models","setup_logging"]

def identity(y):
    return y
def identity_inverse(y):
    return y

class Model:
    '''
    Master modeling class that handles many of the details of Machine Learning.
    Implements the followin generic model flow. However, the pipeline objet 
    is able to be updated prior to fitting.
  
    Model Flow:
        1. get analysis columns only ( throw away rest)
        2. na_preprocess analysis columns (either impute or drop)
        3. calculate X based on patsy formula
        4. add new features (Poly, Power, etc.)
        5. do feature selection
        6. fit chosen model

    Args:
        df (pd.DataFrame): DataFrame for analysis
        formula (str): Formula string based on patsy. Can use "`" to get all variables
        model (model | list of models): A model that implements model.fit(X,y) and model.predict(X)
        test_split (float): percentage of data points to be held out from fitting algorithm
        random_state (int): passed for test split and any algorithm that accepts random_state
        handle_na (bool): if True fills NA using an algorithm. If False drops NAs
        preprocessor (sklearn.pipeline): default used Model Flow. However, can replace with user define algorithm
        title (str): Name of the analysis used for documentation
        analyst (str): Name of the analyst used for documentation
        description (str): Description of analysis used for documentation 
        **kwargs: Keywords can also be passed to further document analysis such as Version = 2020 Update Cycle     
    
    Example:
        df = pd.DataFrame({'y': [1,2,3,4,5], 'x1': [2,4,6,8,10], 'x2': ["a", "b","b","a","a"]})
        myModel = Model(df, "y~x1+x2-1", model= LinearRegression(),
            meta_data={
                'title': "Example Analysis",
                'desc': "Do some anlaysis"}
                )
        myModel.fit().summary()
        myModel.predict(pd.DataFrame()) # use an empty dataframe to predict mean estimate
        myModel.save("myModel")
        app = myModel.report(show=False)
        app.save


        # load data
        loadedModel = Model.load("myModel")
    
    '''

    def __init__(self, df=None, formula=None, target=None, model=RandomForestRegressor(),y_transform=None, y_inverse = None, test_split=.2, random_state=42, handle_na=True,preprocessor=True, title="Generic Report Title", description="N/A", analyst="N/A", **kwargs):
        # Get attributes
        logger.info(f"Initializing Model with title: {title}")
        logger.debug(f"Model parameters: formula={formula}, target={target}, model={model}, test_split={test_split}")
        
        self._meta_data = dict(title = title,description= description,analyst = analyst, **kwargs)
        #self._meta_data = dict(**meta_data, **kwargs)
        
        
        # TODO: Solve for y_transform and y_inverse based on formula provided
        if y_transform is None:
            self.y_transform = identity
        else:
            self.y_transform = y_transform
        if y_inverse is None:
            self.y_inverse = identity_inverse
        else:
            self.y_inverse = y_inverse
        
        
        
        # allow user to load data from file
        if df is None:
            logger.info("No dataframe provided, opening data selection dialog")
            df = self.open_data()

        # allow user to enter either formula or target variable
        if (formula is None):  # Start Feature selection routine
            if (target is None):
                logger.info("No formula or target provided, prompting user for target column")
                print(df.columns.tolist())
                target = input("Choose Target Column: ")  # df.columns[0]
                if target in ["quit", "exit", "q", "e"]:
                    logger.info("Exiting program")
                    sys.exit()
                logger.info(f"User selected target column: {target}")

            # Start Feature selection routine
            # Implement pipeline to Add Features / Remove Features
            formula = f"Q('{target}') ~ ` "
            logger.info(f"Auto-generated formula: {formula}")
            
        try:
            logger.debug("Parsing formula wildcard")
            self.formula = process.MakeFormula.parse_formula_wildcard(formula, df)
            logger.debug(f"Parsed formula: {self.formula}")
            
            logger.debug("Getting formula columns")
            self.analysis_cols = self.get_formula_cols(formula, df)
            self.target_cols = self.get_formula_cols(formula, df, target_val=True)
            self.feature_cols = self.get_formula_cols(formula, df, feature_vals=True)
            
            logger.debug(f"Target columns: {self.target_cols}")
            logger.debug(f"Feature columns: {self.feature_cols}")
            logger.debug(f"Analysis columns: {self.analysis_cols}")

            self.df = df[self.analysis_cols]
            self.handle_na = handle_na
            self.formula = formula
            self.ModelDate = datetime.now()
            
            # Test Train Split
            self.random_state = random_state
            self.test_split = test_split

            # get y, X to fit data on
            logger.info("Creating target variable using MakeFormula")
            self.y = process.MakeFormula(self.formula, handle_na=self.handle_na,return_X=False,return_y=True, return_type='dataframe').fit_transform(self.df)
            
            # check transform
            logger.debug("Validating transformation functions")
            # get absolute percentage error
            transform_test_vals = np.abs(self.y_inverse(self.y_transform(self.y)) / self.y -1)
            if transform_test_vals.sum()[0] >= .0001: 
                logger.warning("Transform function validation failed - y_transform and y_inverse may not be correct")
                logger.warning(f"Average absolute error: {np.mean(transform_test_vals)}")
                print("WARNING!!!! y transform and y_inverse are not correct")
                print("Average Abs Error",np.mean(transform_test_vals))

            self.X = self.df
            logger.info("Performing train-test split")
            if (test_split > 0.0) & (test_split < 1.0):
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X, self.y, test_size=self.test_split, random_state=self.random_state)
                logger.debug(f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}")
            else:
                logger.info("No train-test split requested, using full dataset for both")
                self.X_train, self.X_test, self.y_train, self.y_test = (
                    self.X, self.X, self.y, self.y)

            # Preprocessor:
            self.preprocessor = preprocessor
            # All data should be numeric by now
            if self.preprocessor is True:
                logger.info("Creating default preprocessing pipeline")
                self.preprocessor = Pipeline(
                    steps=[
                        ('dateCleaner', process.DateTransform()),
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                        #('addPoly', PolynomialFeatures(include_bias=False) )
                    ]
                )
                
            logger.info("Creating model pipeline")
            self.model = Pipeline([
                ('CheckFeatures',process.FeatureCheck() ), # adds features to new models if missing
                ('formula', process.MakeFormula(self.formula, handle_na=self.handle_na, return_X=True, return_y=False) ),
                ('preprocess', self.preprocessor), # implement rest of pipeline. All features should be numeric now
                ('model', model)]) # implement estimator
            
            logger.info("Model initialization complete")
            
        except Exception as e:
            logger.error(f"Error during model initialization: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            raise

    def __repr__(self):
        summary = ("Model Summary\n"
                   f"  Formula: {self.formula}\n"
                   f"  Model:{self.model.__str__()}\n"
                   )

        try:
            results = self.summary()
            #summary += f"\n{results}"
            # print(results)

        except:
            summary += "\n\nModel has not been fit to data yet"

        return summary
    def __getattr__(self, name):
        '''Treat this class as a the underlying model object'''

        if name != 'model':
            return getattr(self.model, name)
        else:
            raise AttributeError(f"Model object has no attribute '{name}'")
    def __getstate__(self):
        '''Pickle Instructions'''
        # nothing to do here anymore since patsy is handled in process module
        #self.y =None
        #self.X =None
        self.save_date = datetime.now()
        #print("I'm being pickled")
        return self.__dict__
    
    # Unpickle
    def __setstate__(self, d):
        '''Unpickle Instructions'''
        self.__dict__ = d
        #self.y,self.X = patsy.dmatrices(self.formula, self.df)

    def fit(self, X=None,y=None, **fit_params):
        try:
            logger.info("Fitting model")
            if X is None: 
                X = self.X_train
                logger.debug("Using default X_train data")
            if y is None: 
                y = self.y_train
                logger.debug("Using default y_train data")

            start_time = timeit.default_timer()
            # start fit routine
            logger.debug(f"Starting fit with parameters: {fit_params}")
            
            self.model = self.model.fit(X, self.y_transform(y))
            
            # end fit routine
            self.run_time = timeit.default_timer() - start_time
            logger.info(f"Model fit completed in {self.run_time:.4f} seconds")
            
            return self
        except Exception as e:
            logger.error(f"Error during model fitting: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            raise

    def predict(self, df=None, X=None):
        try:
            logger.info("Making predictions")
            
            if df is None:
                if X is None:
                    logger.debug("Using default X data for prediction")
                    X = self.X
                else:
                    logger.debug("Using provided X data for prediction")
            else:
                logger.debug("Using provided DataFrame for prediction")
                X = df.copy()
            
            logger.debug(f"Prediction input shape: {X.shape}")
            result = self.y_inverse(self.model.predict(X))
            logger.debug(f"Prediction completed, output shape: {result.shape if hasattr(result, 'shape') else 'scalar'}")
            
            return result
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            raise

    

    def get_transformed_x(self, X=None):
        if X is None: X = self.X
        if isinstance(self.model, Pipeline):
            return self.get_pipeline(self.model).transform(X)
        else:
            return X

    
    @staticmethod
    def get_pipeline(pipe):
        if isinstance(pipe, Pipeline):
            new_pipe = []
            for step in pipe.steps:
                if hasattr(step[1], 'transform'):
                    new_pipe.append((step[0], step[1]))
                else:
                    hasattr(step[1], 'predict')
                    return Pipeline(new_pipe)
            return Pipeline(new_pipe)


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
            'RMSE':[metrics.mean_squared_error(y_pred, y_test)**.5] ,
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

            }).assign(TrainY_CV= lambda x: x.TrainY_STD / x.TrainY_Mean )
        else:
            Train_Info = pd.DataFrame()

        results = pd.concat([results, Train_Info], axis=1)

        return results

    def summary(self):

        # s = self.stats(
        #     model=self.model,
        #     X_test=self.X_test,
        #     y_test=self.y_test,
        #     X_train=self.X_train,
        #     y_train=self.y_train
        # )

        # results = s.assign(
        #     Model=[self.model['model']],
        #     Formula=self.formula,
        #     RunTime=self.run_time,
        #     ModelDate=self.ModelDate
        # )
        def calculate_cv(y_true, y_pred):
            """Calculates the coefficient of variation for a regression model.

            Args:
                y_true: The true target values.
                y_pred: The predicted values.

            Returns:
                The coefficient of variation.
            """
            try:
                residuals = y_true - y_pred
                mean_y_true = np.mean(y_true)
                std_dev_residuals = np.std(residuals)
                cv = std_dev_residuals / mean_y_true
                return float(cv)
            except:
                return np.nan
        y_test = self.y_test
        y_pred = self.model.predict(self.X_test)
        y_pred_train = self.model.predict(self.X_train)

        results = pd.DataFrame({
            'Model': [self.model['model']],
            'Formula': [self.formula],
            'RunTime': [self.run_time],
            'ModelDate': [self.ModelDate],
            'ReportDate': [datetime.now()],
            'TestRSQ': [metrics.r2_score(y_test, y_pred)],
            'TestMSE': [metrics.mean_squared_error(y_test, y_pred)],
            'TestAbsErr': [metrics.mean_absolute_error(y_test, y_pred)],
            'TestCV': [calculate_cv(y_test, y_pred)],
            'TrainDF': [self.X_train.shape[0] - self.X_train.shape[1]],
            'TestDF': [self.X_test.shape[0] - self.X_test.shape[1]],
            'TestMaxError': [metrics.max_error(y_test, y_pred)],
            'TrainRSQ': [metrics.r2_score(y_pred_train, self.y_train)],
            'TrainMSE': [metrics.mean_squared_error(y_pred_train, self.y_train)],
            'TrainAbsErr': [metrics.mean_absolute_error(self.y_train, y_pred_train)],
            'TrainCV': [calculate_cv(self.y_train, y_pred_train)],
            'TrainMaxError': [metrics.max_error(self.y_train, y_pred_train)]
        })

        return results

    def save(self, name, remove_data=False, compress=3):

        if ".joblib" not in name:
            name = name + ".joblib"

        obj = copy.deepcopy(self)
        if remove_data:
            del obj.X
            del obj.y
            del obj.df
            del obj.X_test
            del obj.X_train
            del obj.y_test
            del obj.y_train
        obj.save_date = datetime.now()
        fName = joblib.dump(obj, name, compress)
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

    def report(self, show=False, **kwargs):
        '''
        Generates a PANEL object of interesting Model Stats and Data Info

        can be saved as HTML, PDF

        '''
        def get_df_info(df):
            import io
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            return s            
        pn.extension('tabulator')
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

        preds = X.assign(
            Actual = self.y.values.ravel(), 
            Predicted = self.predict(X=self.X).ravel(),
            Residual = lambda x: x.Actual - x.Predicted
            )
        act_vs_pred = preds.hvplot(x='Predicted', y='Actual', kind='scatter',
                                   title='Actual vs Predicted') * hv.Slope(1, 0).opts(color='red')
        residual_plot = preds.hvplot(x='Predicted', y='Residual', kind='scatter', title='Residuals') * hv.Slope(0, 0).opts(color='red')
        # Plot X vs Y predicted and actual for each column in X
        selected_col = pn.widgets.Select(name='Select Column',value=X.columns[0], options=X.columns.tolist())
        @pn.depends(selected_col)
        def plot_col(col):
            return preds.hvplot(x=col, y='Actual', kind='scatter',color='black', title=f'{col} Actual vs Predicted')* preds.hvplot(x=col, y='Predicted', kind='scatter', color='blue', title='Predicted')

        summary = pn.Row(
            pn.Card(summary_df, title='Summary Statistics', height=500),
            pn.Column(
                pn.Card(
                    pn.Tabs(
                        ('Actual Vs Predicted', act_vs_pred),
                        ('Residuals', residual_plot),
                        ('Column vs Actual',pn.Column(selected_col, plot_col))
                    ),
                    title="Actual Vs Predicted",
                    height=500
                    )
            )
        )

        raw_data_page = pn.Row(
            pn.Card(self.df, title='Data Info'),
            pn.Card(self.df.describe(), height=500, title='Data Stats')
        )
        pipeline_transformations = self.show_pipeline_transformations()
        t_tabs=pn.Tabs()
        for name, processor, transformed_df in pipeline_transformations:
            t_tabs.append(
                (name, pn.Column(
                    processor.__repr__(), 
                    pn.Card(transformed_df, height=500, title=name)
                ))
            )
    
        transformed_data_page = pn.Row(
            #pn.Card(df_transformed, title='Transformed Data Info'),
            pn.Card(t_tabs, title='Transformed Data', height=500)
        )



        pages = pn.Tabs(('Summary', summary),
                        ('Raw Data Stats', raw_data_page),
                        ('Transformed Data', transformed_data_page)
                        #,('Feature Importance', pn.panel("in work"))
                        )

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
    def find_column_names(model, X):
        coef_names = None
        n = len(model) -1
        for i in range(n):
            try:
                tmp_pipe = model[0:n-i]
                if isinstance(tmp_pipe,ColumnTransformer):
                    print('need to implmeent columntransformer')
                    coef_names = [*[p.get_features_names() for p in tmp_pipe]]
                else:
                    coef_names = tmp_pipe.transform(X).columns.to_list()
    
                return coef_names

            except:
                pass
        return coef_names
    
    def model_coefs(self):
        return self.get_coefs(self.model, self.X)

    @staticmethod
    def get_coefs(model=None, X=None):        
        '''
        Given a pipeline model attempt find column names and coefs

        PARAMS:
            model (sklearn.pipeline): model object
            X (pd.DataFrame): dataframe used to train model
        '''
        coef_names = Model.find_column_names(model, X)
        try:
            # make generic model[len(model)].coef_
            regressor  = model[len(model)-1]
            coefs = [regressor.intercept_, *regressor.coef_]
            coef_df = pd.DataFrame(columns =['Intercept', *coef_names])
            df.loc[0] = coefs 
            return coef_df
        except:
            print('could not find coefficients')
            return pd.DataFrame()

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
        df = df.sample(2)
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

    def show_pipeline_transformations(self, n_rows=5):
        '''
        Show the transformations applied to the data in the pipeline
        PARAMS:
            n_rows: Number of rows to use for transformation (default: 5) if -1 all rows are used
        '''
        return Model.pipeline_transformations(self.model, self.df, n_rows=n_rows)


    @staticmethod
    def pipeline_transformations(pipeline, df,n_rows=5):
        """
        Recursively steps through each transformation in a pipeline and returns steps with transformed data.
        
        Args:
            pipeline: An sklearn Pipeline object
            df: Input DataFrame for transformation
            n_rows: Number of rows to use for transformation (default: 5) if -1 all rows are used
            
        Returns:
            List of tuples: (step_name, step_object, transformed_data) for each step
        """
        results = []
        if n_rows == -1:
            tmp_df = df.copy()
        else:
            tmp_df = df.copy().head(n_rows)
        
        for name, processor in pipeline.steps:
            # Handle nested pipelines
            if hasattr(processor, 'steps'):  # Is a Pipeline or similar
                nested_results = Model.pipeline_transformations(processor, tmp_df)
                # Add nested step name prefix
                results.extend([(f"{name}_{sub_name}", sub_processor, sub_df) 
                               for sub_name, sub_processor, sub_df in nested_results])
            else:
                # Apply transformation if possible
                if hasattr(processor, 'transform'):
                    transformed = processor.transform(tmp_df)
                    
                    # Convert to DataFrame if not already
                    if not isinstance(transformed, pd.DataFrame):
                        try:
                            # Try to use original column names
                            # TODO: add set_output(transform="pandas") for transformers.
                            transformed_df = pd.DataFrame(transformed, 
                                                          columns=tmp_df.columns)
                        except:
                            # Fallback to generic column names
                            transformed_df = pd.DataFrame(transformed)
                    else:
                        transformed_df = transformed
                    
                    # Update tmp_df for next step
                    tmp_df = transformed_df
                    
                    results.append((name, processor, transformed_df))
                elif hasattr(processor, 'predict'):
                    tmp_df = processor.predict(tmp_df)
                    if not isinstance(tmp_df, pd.DataFrame):
                        tmp_df = pd.DataFrame(tmp_df, columns = ['predicted'])
                    results.append((name, processor, tmp_df))
                else:
                    pass
        
        return results

class LC_Model(Model):
    '''
    Notes:
        

    '''
    learn_formula = 'np.log(lot_midpoint)'
    rate_formula = 'np.log(lot_qty)'
    def __init__(self, df,x_formula='',AUC_col='auc',quantity_column='Qty', lot_order_cols=['FY'], grp_cols=[],priors_column='priors', model=RidgeCV(alphas=np.arange(0.0001,.2, .001)), First=None, Last=None, unit_no=None,y_transform=np.log, y_inverse=np.exp,  **kwargs):
        
        logger.info(f"Initializing LC_Model with AUC_col={AUC_col}, quantity_column={quantity_column}")
        logger.debug(f"LC_Model parameters: lot_order_cols={lot_order_cols}, grp_cols={grp_cols}")
        
        try:
            super().__init__(df=df,formula=f"{AUC_col} ~ `", model=model,y_transform=np.log, y_inverse=np.exp,**kwargs)

            logger.info("Creating LC preprocessing pipeline")
            prep_LC = process.LC_Lot_Midpoint(meta_columns=grp_cols, lot_order_columns=lot_order_cols, quantity_column=quantity_column, priors_column=priors_column, lc_slope=1)
            
            # Construct formula based on inputs
            if x_formula == "":
                new_formula = f"{AUC_col} ~  {self.learn_formula} + {self.rate_formula} -1"
            elif x_formula[0] in (":", "*", "-", "+"):
                new_formula = f"{AUC_col} ~  {self.learn_formula} + {self.rate_formula}{x_formula} -1"
            else:
                new_formula = f"{AUC_col} ~  {self.learn_formula} + {self.rate_formula}+ {x_formula} -1"
            
            logger.debug(f"Using formula: {new_formula}")
            
            # Insert the LC preprocessing step into the pipeline
            logger.debug("Inserting LC preprocessing into pipeline")
            self.model.steps.insert(1,('prep_LC', prep_LC))
            self.model.set_params(formula__formula = new_formula)
            
            # Setup the transformed target regressor
            logger.debug("Creating transformed target regressor")
            self.lc_model = TransformedTargetRegressor(self.model, func=np.log, inverse_func=np.exp)

            # Not very elegant, but need to reassign df, y, X 
            self.df = df

            # Get y, X to fit data on
            self.X = self.df
            if (self.test_split > 0.0) & (self.test_split < 1.0):
                logger.debug("Performing train-test split for LC model")
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X, self.y, test_size=self.test_split, random_state=self.random_state)
            else:
                logger.debug("Using full dataset for both training and testing")
                self.X_train, self.X_test, self.y_train, self.y_test = (
                    self.X, self.X, self.y, self.y)
                    
            logger.info("LC_Model initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing LC_Model: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            raise

    def fit(self):
        logger.info("Starting LC_Model fit")
        try:
            #self.model.fit(self.X_train, self.y_train)
            start_time = timeit.default_timer()
            slope = 1
            fit_slope = 0
            cnt = 0
            
            logger.info("Iterative fitting of LC model to optimize slope")
            while cnt <= 10:
                cnt = cnt + 1
                logger.debug(f"Iteration {cnt}, current slope: {slope}")
                
                self.model.set_params(prep_LC__lc_slope=slope)
                self.model.fit(self.X_train, np.log(self.y_train))
                
                self.lc_model.regressor.set_params(prep_LC__lc_slope=slope) 
                self.lc_model.fit(self.X_train, self.y_train)
                
                fit_slope = 2**(self.lc_model.regressor_['model'].coef_[0])
                logger.debug(f"Iteration {cnt}: New slope = {fit_slope}")
                print("iteration:", cnt,":", fit_slope)
                
                if np.abs(slope-fit_slope) <= .0000001:
                    logger.info(f"Convergence achieved at iteration {cnt}, final slope: {fit_slope}")
                    break
                else:
                    slope = fit_slope
                    
            self.run_time = timeit.default_timer() - start_time
            logger.info(f"LC_Model fit completed in {self.run_time:.4f} seconds")
            
            return self
            
        except Exception as e:
            logger.error(f"Error during LC_Model fitting: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            raise

    def predict(self, df=None, X=None):
        logger.info("Making LC_Model predictions")
        try:
            if df is None:
                if X is None:
                    logger.debug("Using default X data for prediction")
                    X = self.X
                else:
                    logger.debug("Using provided X data for prediction")
            else:
                logger.debug("Using provided DataFrame for prediction")
                X = df.copy()
            
            logger.debug(f"Prediction input shape: {X.shape}")
            result = np.exp(self.model.predict(X))
            logger.debug(f"Prediction completed, shape: {result.shape if hasattr(result, 'shape') else 'scalar'}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during LC_Model prediction: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            raise

class Models:
    '''
    This class provides an interface to build many different models based on the `Model` class
    and store them in a single place "db". This DataFrame stores each model and it's meta data
    self.db.Model is the colum that stores the actual Model.

    KNOWN ISSUE!!! 
        When building many models this can exceed memory and cause models
        to appear not to fit. In such cases it may be useful to filter db
        to the model of interest and use the Model interface.
    '''
    def __init__(self, df, formulas=[], by=[], target=None, models=[LinearRegression(),
                                                                    RandomForestRegressor(),
                                                                    LassoCV(
                                                                        cv=5),
                                                                    RidgeCV(
                                                                        cv=5),
                                                                    ElasticNetCV(cv=5)], **kwargs):


        # Initialize Models Variables
        self.ModelDate = datetime.now()
        self.Models = {}
        self.db = pd.DataFrame()
        self.data = [df]
        self.results = pd.DataFrame()
        self.n_models = 0
        self.run_time = dt.datetime.now() - dt.datetime.now()
        self.by = by

        # get all args that were passed
        args = locals()
        del args['self']
        del args['kwargs']

        # Build Models and Add Them to DB
        self.build_models(**args, **kwargs)

    def __repr__(self) -> str:
        s = "Many Models API\n"
        s+= f"{'Title: '.join(self.db['title'].unique())}"
        s+= f"{len(self.db)} Models were fit"
        
        return s

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
                                                                        ElasticNetCV(cv=5)], **kwargs):

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
                        tmp_meta = []
                    else:
                        tmp_by = by
                        tmp_meta = meta
                    
                   

                    self.Models[i] = Model(
                        df=frame, formula=f, model=mod, **kwargs, 
                        **{col: meta for col, meta in zip(by, *[tmp_meta] if len(tmp_meta)>1 else [tmp_meta])}
                        )
                    _model_specs[i] = dict(
                        **{**self.Models[i]._meta_data, **kwargs},
                        #**kwargs,
                        Formula=f,
                        ModelType=mod.__repr__(),
                        Model=self.Models[i],
                        Target=self.Models[i].target_cols,
                        Features=self.Models[i].feature_cols,
                        AnalysisColumns=self.Models[i].analysis_cols,
                        BY=tmp_by,
                        BY_META = [*tmp_meta]
                        #**{col: meta for col, meta in zip(tmp_by, *[tmp_meta] if len(tmp_meta)>1 else [tmp_meta] )}
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
        if by is None:
            by = self.by

        for meta, frame in df.groupby(by):
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

    def report(self, show=False, **kwargs):
        #X = self.X
        app = pn.template.BootstrapTemplate(title="Model Report")
        pn.config.sizing_mode = "stretch_width"
        pn.extension('tabulator')
        # Header
        app.header.header_background = 'blue'

        # Side Bar
        # inputs = dict()
        # for col in self.db.Features[1]:
        #     if self.df[col].dtype == 'object':
        #         inputs[col] = pn.widgets.Select(
        #             name=col, value=self.df[col][0], options=self.df[col].unique().tolist())
        #     else:
        #         inputs[col] = pn.widgets.FloatSlider(
        #             name=col, value=self.df[col].median())

        # widgets = pn.WidgetBox("# Model Inputs", *[inputs[w] for w in inputs])
        # app.sidebar.append(widgets)
        
        # Main
        db = self.db.drop('Model', axis=1)
        summary_df = self.summary()
        #summary_df.columns = summary_df.loc[0]
        #summary_df = summary_df.loc[1:]
        #preds =pd.DataFrame({"Actual" : self.df[self.Target[0]], "Predicted" : self.predict()})
        #act_vs_pred = preds.hvplot(x='Predicted', y='Actual',kind='scatter', title='Actual vs Predicted') * hv.Slope(1,0).opts(color='red')
        summary = pn.Row(
            pn.Card(db, height=500, title='Available Models'),
            pn.Card(summary_df, height=500, title='Summary Statistics')
            #pn.Card(act_vs_pred, title= "Actual Vs Predicted" , height=500)
        )
        pages = pn.Tabs(('Summary', summary)
                        #('Feature Importance', pn.panel("in work"))
                        )

        
        # Create a selection widget for the models
        model_names = {f"{index} ; {row.Model.model[-1].__class__.__name__} ; {row.Formula} ; {','.join([by+'='+meta for by,meta in zip(row.BY, row.BY_META)])}": index for index, row in self.db.iterrows()}
        #print(model_names)
        model_select = pn.widgets.Select(name='Model', options=model_names)
        
        @pn.depends(model_select)
        def get_model(index):
            #print(index)
            index = index
            model = self.db.iloc[index].Model
            try:
                report = model.report(show=False)
                
            except:
                report = model.fit().report(show=False)
            return pn.Column(*report.main, sizing_mode="stretch_width")
        pages.append(("Models", pn.Column(model_select, get_model, sizing_mode="stretch_width")))
        
        app.main.append(pages)



        if show:
            server = app.show("KJ's Model Anlaysis report", threaded=True)
            return (app, server)
        else:
            return app

    def summary(self):
        #s = [mod.fit().summary() for mod in self.db.Model]

        results = []    
        for index, row in self.db.iterrows():
            mod = row.Model
            try:
                try:
                    summary = mod.summary()
                except:
                    summary = mod.fit().summary()
                summary.index = [index]
            except:
                summary = pd.DataFrame({'FitError':[True]}, index=[index])
            results.append(summary)
        return pd.concat(results) 

class LC_Models:
    pass
if __name__ == "__main__":
    import pycost as ct
    # Learning Curve Example
    # Create learning curve data
    def make_lc_data(n_groups=5, n_units=20,  unit_slope=0.95, rate_quantity_slope=None):
        df = pd.DataFrame()
        for i in range(n_groups):
            group_data = pd.DataFrame({
                'group': ['program '+str(i)]*n_units,
                'FY': np.arange(2010,2010+n_units),
                'unit': np.arange(1,1+n_units),
                'rate_quantity': [n_units]*n_units,
                'priors': [0]*n_units
            }).assign(value = lambda x: np.random.normal(100, 10) * (x.unit ** np.log(unit_slope)/np.log(2)) * (x.rate_quantity ** np.log(rate_quantity_slope)/np.log(2)))
            df = pd.concat([df, group_data])
        return df
    
    df = make_lc_data(n_groups=5, n_units=20, rate_quantity_slope=0.95, unit_slope=0.95)
    print(df)
    lc_model = LC_Model(df, 
                        AUC_col='value',quantity_column='unit',lot_order_cols=['FY'], grp_cols=['group'], priors_column='priors',
                        test_split=0)
    lc_model.fit()
    print(lc_model.summary())


    # fit multiple models
    df2 = ct.analysis.process.LC_Lot_Midpoint.lc_prep(df, lot_order_columns=['group', 'FY'], quantity_column='unit', priors_column='priors')

    print(df2.head())

    lc_model2 = Models(df2, 
                       formulas = ["value ~ np.log(unit)", "value ~ np.log(unit) + np.log(lot_qty)"], # learn and learn and rate
                       by=['group'], # fit each group separately
                       models=[RidgeCV(alphas=np.arange(0.0001, 0.2, 0.001))], # use ridge regression
                       y_transform=np.log, # log transform the y variable
                       y_inverse=np.exp, # inverse transform the y variable
                       test_split=0) # don't split the data
    

    lc_model2.fit()
    print("starting model summary")
    print(lc_model2.db.drop(columns=['Model']).to_csv("lc_model2_summary.csv"))
    #print(lc_model2.predict(df2))
    for model in lc_model2.db.Model:
        model.fit()
        df2['predicted_'+model.formula] = model.predict(df2)


    """
    # Model Example For predicting inflation
    df = ct.inflation.jic
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

    # Show transformation
    Model.show_transformation(m, df)"
    """
