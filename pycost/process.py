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

__all__=["AutoPreProcess", "DateTransform", "FeatureCheck", "ImputeNA", "MakeFormula", "Clean"]

def iferror(func,*args, **kwargs):
    if "error" not in kwargs: 
        error=None
    else:
        error = kwargs["error"]
        kwargs.pop('error')
    try:
        if len(args) > 0 and len(kwargs) ==0:
            results = func(*args)
        elif len(args) ==0 and len(kwargs) >0:
            results = func(**kwargs)
        elif len(args) > 0 and len(kwargs) > 0:
            results = func(*args, **kwargs)
        else:
            func() 
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

    def __init__(self,date_columns=[],drop=True, cont_year=True,year=True,month=True,day=False,weekday=False,season=False,**kwargs):
        self.date_columns =date_columns
        self.drop = drop
        self.cont_year =cont_year
        self.year=year
        self.month=month
        self.day=day
        self.weekday=weekday
        self.season =season

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
                if self.weekday: X[f"{col}_weekday"] = tmp_date.dt.weekday
                if self.season: X[f"{col}_season"] = tmp_date.map(self.season_of_date)
                if self.drop: X.drop(col, axis=1, inplace=True)
            except:
                print(f"{col} could not complete")

        return X

    @staticmethod
    def season_of_date_column(date_col):
        return date_col.map(self.season_of_date)

    @staticmethod
    def season_of_date(date):
        try:
            year = str(date.year)
            seasons = {
                'spring': pd.date_range(start='03/21/'+year, end='06/20/'+year),
                'summer': pd.date_range(start='06/21/'+year, end='09/22/'+year),
                'fall': pd.date_range(start='09/23/'+year, end='12/20/'+year),
                }
            if date in seasons['spring']:
                return 'spring'
            if date in seasons['summer']:
                return 'summer'
            if date in seasons['fall']:
                return 'fall'
            else:
                return 'winter'
        except:
            return np.nan

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
        self.add_features=add_features
        self.coerce_type=coerce_type

    
    def fit(self,X,y=None):
        df = pd.DataFrame(X).copy()
        self.columns = df.columns.to_list()
        self.dtypes = df.dtypes
        self.sample_data = df.head()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        cols = X.columns.tolist()
        add_cols = set(self.columns) - set(cols)
        if add_cols == set(self.columns): X = pd.DataFrame({self.columns[0]: [np.nan]})
        if self.add_features:
            
            X[list(add_cols)] = np.nan
        else:
            print(f"failed feature check",f"{len(add_cols)} featuers not present in dataset", end="/n")
            
        if self.coerce_type:
            for col in self.columns:
                X[col] = X[col].astype(self.sample_data[col].dtype, errors="ignore")

        return X

class ImputeNA(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder."""

    def __init__(self,numeric_imputer=SimpleImputer( strategy='median'), categorical_imputer=SimpleImputer(strategy='most_frequent'),**kwargs):
        self.numeric_imputer =numeric_imputer
        self.categorical_imputer = categorical_imputer
    def fit(self, X:pd.DataFrame, y=None):
        X = X.copy()
        self.columns = X.columns.tolist()
        self.num_cols = X.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        if len(self.num_cols) > 0: self.numeric_imputer=self.numeric_imputer.fit(X[self.num_cols])
        if len(self.cat_cols) >0: self.categorical_imputer=self.categorical_imputer.fit(X[self.cat_cols])
        
        self.rest_cols = set(self.columns) - set(self.num_cols) - set(self.cat_cols)


        return self

    def transform(self, X:pd.DataFrame):
        X = X.copy()
        cols = X.columns.tolist()
        add_cols = set(self.columns) - set(cols)
        X[list(add_cols)] = np.nan


        if len(self.num_cols) >0: 
            nums = self.numeric_imputer.transform(X[self.num_cols])
            nums = pd.DataFrame(nums, columns=self.num_cols)
        else:
            nums = pd.DataFrame()

        if len(self.cat_cols) >0: 
            cats = self.categorical_imputer.transform(X[self.cat_cols])
            cats = pd.DataFrame(cats,columns = self.cat_cols)
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
    formula: default "`" wildcard string to get all variables (this is exclusive to this libary)
        for more details see patsy
    
    handle_na: default =ImputeNA()
        patsy default is to get rid of NAs. However this defaults to keep NA's by IMputing
        pass your own Handle NA function with fit and transform methods (must return DataFrame)
        pass None or False to remove rows with NA's
    
    return_type: default = 'dataframe'
        Either "matrix" or "dataframe"
    
    return_X default =True

    return_y default =False
    
    """
    
    def __init__(self,formula='`', handle_na=ImputeNA(),return_type='dataframe',keep_cols="all", return_X=True,return_y=False):
        self.formula = formula
        
        self.handle_na=handle_na
        if self.handle_na==False: self.handle_na = None
        if self.handle_na==True: self.handle_na = ImputeNA()

        self.return_type = return_type
        self.keep_cols = keep_cols
        self.return_X = return_X
        self.return_y = return_y

    
    def __getstate__(self):
        
        '''Pickle Instructions'''
        self.y =None
        self.X =None
        #self.save_date = datetime.now()
        #print("I'm being pickled")
        return self.__dict__
    
    def __setstate__(self, d):
        '''Unpickle Instructions'''
        self.__dict__ = d
        if not self.df.empty():
            self = self.fit(self.df)
    
    def fit(self, X, y=None):
        # handle NAs
        X = X.copy()
        if not self.handle_na is None: self.handle_na.fit(X)

        # replace wildcard in formula
        self.formula = self.parse_formula_wildcard(self.formula, X, wildcard="`")
                
        # parse formula
        self.split_formula = self.formula.split("~")
        
        # keep copy of data
        self.df = X

        # find design matrix formula
        if len(self.split_formula) > 1:
            self.y, self.X = patsy.dmatrices(self.formula, self.df)
            self.y=self.y.design_info
            self.X = self.X.design_info
        else:
            self.X = patsy.dmatrix(self.formula, self.df)
            self.X = self.X.design_info
        

        # fin

        return self

    @staticmethod
    def parse_formula_wildcard(formula,df, wildcard="`"):
        # find wildcard in formula
        
        if "~" in formula:
            target = MakeFormula.get_formula_cols(formula, df, target_val=True, feature_vals=False)
        else:
            target = []
        if wildcard in formula:
            # add check to see if variable can be added
            all_cols = []
            not_added = []
            for col in [f"{col}" for col in df.columns if not col in target ]:
                try:
                    tmp = patsy.dmatrix(col, df)
                    all_cols.append(col)
                except:
                    not_added.append(col)
            all_cols = " + ".join(all_cols)
            formula=formula.replace(wildcard, all_cols)
        
        return formula

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


    def transform(self, X):
        X = X.copy()
        if not self.handle_na is None: X = self.handle_na.transform(X)

        if self.return_X & self.return_y:
            X_transform = patsy.build_design_matrices([self.X], X, return_type=self.return_type)[0]
            y_transform = patsy.build_design_matrices([self.y], X, return_type=self.return_type)[0]

            ans = (y_transform, X_transform)
        
        elif self.return_X:
            X_transform = patsy.build_design_matrices([self.X], X, return_type=self.return_type)[0]
            ans = X_transform
        
        elif self.return_y:
            y_transform = patsy.build_design_matrices([self.y], X, return_type=self.return_type)[0]
            ans = y_transform
        
        else:
            raise ValueError(self, "Need to choose an return X or return Y")

        return ans

class LC_Lot_Midpoint(BaseEstimator, TransformerMixin):
    '''
    routine to automatically calculate First Unit, Last Unit, MidpointQty
    '''

    def __init__(self, meta_columns=[], lot_order_columns=['FiscalYear'], quantity_column='value', priors_column = None, lc_slope=1, lot_qty_col ='lot_qty', lot_midpoint_col='lot_midpoint'):
        self.meta_columns = meta_columns
        self.lot_order_columns = lot_order_columns
        self.quantity_column = quantity_column
        self.priors_column = priors_column
        self.lc_slope = lc_slope
        self.lot_qty_col = lot_qty_col
        self.lot_midpoint_col = lot_midpoint_col

    def fit(self, X,y=None):

        # Nothing to do...maybe store priors?
        
        return self
    

    def transform(self,X):
        df = X.copy()
        df = self.lc_prep(
            df = df,
            cols = [*self.meta_columns, *self.lot_order_columns], 
            val=self.quantity_column,
            priors_column= self.priors_column,
            lc_slope = self.lc_slope,
            lot_qty_col=self.lot_qty_col,
            lot_midpoint_col=self.lot_midpoint_col ) 
        return df

    @staticmethod
    def lc_prep(df, cols, val= "value",priors_column=None, lc_slope=1, lot_qty_col='lot_qty', lot_midpoint_col='lot_midpoint'):
        

        if pd.__version__ >= '1.0':
            lc =df.groupby(cols)[val].agg(share_qty = 'sum').reset_index()
            lc = lc.rename(columns = {'share_qty': lot_qty_col})
        else:
            lc = df.groupby(cols)[val].agg(sum).reset_index()
            lc = lc.rename(columns={val:lot_qty_col})
        if len(cols) >1:
            lc["Last"] = lc.groupby(cols[:-1])[lot_qty_col].cumsum()
        else:
            lc['Last'] = lc[lot_qty_col].cumsum()
        lc["First"] = lc["Last"] - lc[lot_qty_col] + 1
        lc[lot_midpoint_col] = np.nan # wait to calculate until we havea priors columns

        lc = lc[cols+ ['First', 'Last', lot_midpoint_col, lot_qty_col]]
        lc = pd.merge(df,lc,how='left', on=cols,sort=False, suffixes=("_orig", ""))

        if priors_column is None:
            priors = 0
        else:
            priors = lc[priors_column]

        lc['Last'] = lc['Last'] + priors
        lc['First'] = lc['First'] + priors
        lc[lot_midpoint_col] = LC_Lot_Midpoint.lc_midpoint(lc["First"], lc["Last"], lc_slope)

        return lc

    @staticmethod
    def lc_midpoint(first,last, lc_slope):
        b = np.log(lc_slope)
        if b == 0:
            return (first+last + 2 *(first*last)**.5)/4
        else:
            midpoint = ((1 / (last - first + 1)) * ((((last + 0.5) ** (1 + b)) - ((first - 0.5) ** (1 + b))) / (1 + b))) ** (1 / b)
            return midpoint


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
