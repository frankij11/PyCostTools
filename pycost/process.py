import pandas as pd
import numpy as np
import datetime

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, PolynomialFeatures, PowerTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

import patsy

def iferror(func,*args, **kwargs):
    if "error" not in kwargs: 
        error=None
    else:
        error = kwargs["error"]
        kwargs.popitem('error')
    try:
        results = func(*args)
    except:
        results = error

    return results

class AutoPreProcess(BaseEstimator, TransformerMixin):
    '''
    Master Level Processor
    '''
    def __init__(self,X,y=None,test_pct=0, preprocess_na=True, applyformula=None, addfeatures=True, selectfeatures=True, **kwargs):
        '''

        PARAMETERS
        ----------

        Set parameters = None if you want to by pass step
        Set parameters = True if you want generic pipeline
        Set parameters = Pipeline Object if you want to use your own pipeline
        '''
        if preprocess_na==True: 
            preproccess_na = self.preprocess()
        elif preprocess_na==False:
            preprocess_na =None
        
        if not applyformula is None:
            applyformula=MakeFormula(applyformula)



        self.pipeline = self.build_pipeline(
            self,
            process_nas=preprocess_na,
            

            **kwargs)
        self.grid_values = None
        return self

    def build_pipeline(self, **kwargs):
        '''
        Keywords are the name of processor steps

        '''

        steps = []
        for key in kwargs:
            if not kwargs[key] is None:
                steps.append((key, kwargs[key]))
        
        self.pipeline = Pipeline(steps)

        pass

    def fit(self,X,y=None):
        '''
        Fit entire pipeline
        '''
        self.columns = X.columns.to_list()
        self.pipeline.fit(X,y)

        return self
    
    def transform(self,X):
        '''
        given fitted pipeline transform data for model
        '''
        X = X.copy()
        # check X against original X
        addCols = set(self.columns) - set(X.columns)
        delCols = set(X.columns) - set(self.columns)
        
        if len(addCols) > 0: X[addCols] = np.nan
        if len(delCols) >0 : X.drop(delCols, axis=1, inplace=True)

        fitted_x = self.pipeline.transform(X)
        return fitted_x
    
    def preprocess(self):
        '''
        Process NA values
        '''
        pass

    def addfeatures(self):
        '''
        Add Features
        '''
        pass

    def selectfeatures(self):
        '''
        Find the best features
        '''
        pass

class DateTransform(BaseEstimator, TransformerMixin):
    """Date to numeric columns."""

    def __init__(self,date_columns=[],drop=True, cont_year=True,year=True,month=True,day=False,weekday=False,**kwargs):
        self.date_columns =date_columns
        self.drop = drop
        self.cont_year =cont_year
        self.year=year
        self.month=month
        self.day=day
        self.weekday=weekday

    @staticmethod
    def find_date_columns(df):
        if not isinstance(df, pd.DataFrame): df = pd.DataFrame(df)
        is_datetime=pd.api.types.is_datetime64_any_dtype
        cols = []
        for col in df.columns:
            if is_datetime(df[col]): cols.append(col)
        return cols

    def fit(self, X=None, y=None):
        # Find date columns
        if not isinstance(self.date_columns, list): self.date_columns = [self.date_columns]
        if len(self.date_columns)==0:
            self.date_columns = self.find_date_columns(X)
        else:
            self.date_columns = self.date_columns
        return self

    def transform(self, X):
        X=X.copy()
        for col in self.date_columns:
            try:
                tmp_date = pd.to_datetime(X[col], errors='coerce')
                #year = tmp_date.apply()
            
                if self.cont_year: X[f"{col}_cont_year"] =  tmp_date.dt.year + (tmp_date.dt.month-1)/12 + (tmp_date.dt.day-1)/365
                if self.year: X[f"{col}_year"] = tmp_date.dt.year
                if self.month: X[f"{col}_month"] = tmp_date.dt.month
                if self.day: X[f"{col}_day"] = tmp_date.dt.day
                if self.weekday: X[f"{col}_day"] = tmp_date.dt.weekday
                if self.drop: X.drop(col, axis=1, inplace=True)
            except:
                print(f"{col} could not complete")

        return X

class FeatureCheck(BaseEstimator, TransformerMixin):
    '''
    Checks to make sure all columns are present. If not present. Then add them as NA or Fail
    

    PARAMTERS
    ---------
    add_features: default =True
        if feature is not present. add feature to list.
    coerce_type: default =True
        try to make types equal to origninal dataframe (will ignore if can't be coerced)
    '''
    def __init__(self, add_features=True, coerce_type=True):
        self.add_feaures=add_features
        self.coerce_type=coerce_type
    
    def fit(self,X,y=None):
        df = pd.DataFrame(X).copy
        self.columns = df.columns.to_list()
        self.dtypes = df.dtypes
    
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        cols = X.columns.tolist()
        add_cols = set(self.columns) - set(cols)
        if add_cols == set(self.columns): X = pd.DataFrame([None], columns=self.columns[0])
        if self.add_feaures:
            X[list(add_cols)] = np.nan
        else:
            print(f"failed feature check",f"{len(add_cols)} featuers not present in dataset", end="/n")
            
        if self.coerce_type:
            X.astype(self.dtypes, errors="ignore")

        return X

class ImputeNA(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder."""

    def __init__(self,numeric_imputer=SimpleImputer(strategy='median'), categorical_imputer=SimpleImputer(strategy='constant', fill_value="missing"),**kwargs):
        self.numeric_imputer =numeric_imputer
        self.categorical_imputer = categorical_imputer
    def fit(self, X:pd.DataFrame, y=None):
        X = X.copy()
        self.columns = X.columns.tolist()
        self.num_cols = X.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        if len(self.num_cols) > 0: self.numeric_imputer=self.numeric_imputer.fit(X[self.num_cols])
        if len(self.num_cols) >0: self.categorical_imputer=self.categorical_imputer.fit(X[self.cat_cols])
        
        self.rest_cols = set(self.columns) - set(self.num_cols) - set(self.cat_cols)


        return self

    def transform(self, X:pd.DataFrame):
        X = X.copy()
        cols = X.columns.tolist()
        add_cols = set(self.columns) - set(cols)
        X[list(add_cols)] = np.nan


        if len(self.num_cols) >0: 
            nums = self.numeric_imputer.transform(X[self.num_cols])
            nums = pd.DataFrame(nums, columns=self.numeric_imputer.get_feature_names())
        else:
            nums = pd.DataFrame()

        if len(self.cat_cols) >0: 
            cats = self.categorical_imputer.transform(X[self.cat_cols])
            cats = pd.DataFrame(cats,columns = self.categorical_imputer.get_feature_names())
        else:
            cats = pd.DataFrame()

        rest = X[self.rest_cols]

        new_X = pd.concat([nums, cats, rest], axis=1)[self.columns]

        return new_X

class MakeDataFrame(BaseEstimator, TransformerMixin):
    """numpy array to data frame"""

    def __init__(self,columns=[None]):
        self.columns = columns
    
    def fit(self,X):
        return self

    def transform(self,X):
        if isinstance(X, pd.DataFrame):
            return X
        else:
            return pd.DataFrame(X, columns = self.columns)

class MakeFormula(BaseEstimator, TransformerMixin):
    """
    String formula to to numbers categorical encoder.

    PARAMTERS:
    ----------
    formula: default '''`''' wildcard string to get all variables (this is exclusive to this libary)
        for more details see patsy
    
    handle_na: default =ImputeNA()
        patsy default is to get rid of NAs. However this defaults to keep NA's by IMputing
        pass your own Handle NA function with fit and transform methods
        pass None or False to remove rows with NA's
    
    return_type: default = 'dataframe'
        Either "matrix" or "dataframe"
    
    return_X default =True

    return_y default =False
    
    """

    def __init__(self,formula='`', handle_na=ImputeNA(),return_type='dataframe', return_X=True,return_y=False):
        self.formula = formula
        
        self.handle_na=handle_na
        if self.handle_na==False: self.handle_na = None
        
        self.return_type = return_type
        self.return_X = return_X
        self.return_y = return_y

        
    
    def __getstate__(self):
        '''Pickle Instructions'''
        self.y =None
        self.X =None
        self.save_date = datetime.now()
        #print("I'm being pickled")
        return self.__dict__
    
    def __setstate__(self, d):
        '''Unpickle Instructions'''
        self.__dict__ = d
        if not self.df.empty():
            self.y,self,X = patsy.dmatrices(self.formula, self.df)
    
    def fit(self, X, y=None):
        if not self.handle_na is None: self.handle_na.fit(X)

        if "`" in self.formula:
            if "'" in self.formula:
                all_cols = " + ".join([f'Q("{col}")' for col in X.columns])
            else:
                all_cols = " + ".join([f"Q('{col}')" for col in X.columns])
            self.formula.replace("`", all_cols)
        self.split_formula = self.formula.split("~")
        self.df = X
        if len(self.split_formula) > 1:
            self.y, self.X = patsy.dmatrices(self.formula, self.df)
        else:
            self.X = patsy.dmatrix(self.formula, self.df)

        return self

    def transform(self, X):
        X = X.copy
        if not self.handle_na is None: X = self.handle_na.transform(X)

        if self.return_X:
            X_transform = patsy.build_design_matrices([self.X.design_info], X, return_type=self.return_type)[0]
        if self.return_y:
            y_transform = patsy.build_design_matrices([self.y.design_info], X, return_type=self.return_type)[0]
        
        if self.return_X & self.return_y:
            return [y_transform, X_transform]

        if len(self.split_formula) > 1:
            ans = patsy.build_design_matrices([self.y.design_info, self.X.design_info], X, return_type=self.return_type)
        else:
            ans= patsy.build_design_matrices([self.X.design_info], X, return_type=self.return_type)[0]
        return ans

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y):
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ['target']

        # persist transforming dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            t = temp.groupby([var])['target'].mean().sort_values(
                ascending=True).index
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])
        return X

class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
    #Check if the variables passed are in a list format, if not convert 
    #to list format and assign it to self.variables to be used in later 
    #methods
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.features_unique =dict()
    
    def fit(self, X:pd.DataFrame,y:pd.Series=None):
        #Nothing to do here, just return the dataframe as is
        # Find unique
        for feature in self.variables:
            self.features_unique[feature] = X[feature].unique()
        return self
    
    def transform(self, X:pd.DataFrame):
	      #Fill missing values and return the modified dataframe
        X=X.copy()
        for feature in self.variables:

            X[feature] = X[feature].fillna("Missing")
        return X

class Clean(BaseEstimator, TransformerMixin):
    def __init__(self,n_cats_max=20, drop_cols=list()):
        self.n_cats_max = n_cats_max
        if type(drop_cols) != list: drop_cols=[drop_cols]
        self.drop_cols = drop_cols
        
        

    def fit(self, X:pd.DataFrame, y:pd.Series=None):
        self.columns = set(X.columns.tolist())
        df = pd.concat([X,y], axis=1)
        df.columns = list(X.columns) + ["y"]
        self.cat_vars = X.select_dtypes(include='object').columns.tolist()
        self.num_vars = X.select_dtypes(include=np.number).columns.tolist()
    
        self.cats_ = dict()
        
        for var in self.cat_vars:
            top_vals = df.groupby([var])["y"].sum().sort_values(
                ascending=False).reset_index()
            top_vals = top_vals.loc[range(0,min(self.n_cats_max+1, len(top_vals))),var].unique().tolist() + ["Other"]     
            self.cats_[var] = top_vals
            
        return self

    def transform(self,X:pd.DataFrame):
        X = X.copy()
        
        # Drop columns not a part of original data set
        extra_cols = set(X.columns.tolist()) - self.columns
        X.drop(columns=extra_cols, inplace=True)
        
        # Fill categorical values that are missing
        # Fill categorical values not in original dataset
        for feature in self.cat_vars:
            X[feature] = X[feature].fillna("Other")
            new_cat = ~X[feature].isin(self.cats_[feature])
            X[feature][new_cat] = "Other"


        # fix!!!
        for var in self.cat_vars:
            for cat in self.cats_[var]:
                col_name = var +"_" + cat
                X[col_name] = X[var] == cat
            X.drop(columns=var, inplace=True)
        


        # Clean Numerical Data
        # Not implemented
        for var in self.num_vars:
            X[var] = pd.to_numeric(X[var], errors='coerce')
        
        # Drop Columns
        for col in self.drop_cols:
            try:
                X.drop(columns=col, inplace=True)
            except:
                pass

        return X
