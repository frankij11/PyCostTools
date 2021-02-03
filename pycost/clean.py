import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import patsy


class MakeFormula(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder."""

    def __init__(self,X=None,y=None, formula='', df=pd.DataFrame()):
        self.formula = formula
        self.y, self.X = patsy.dmatrices(formula, df,return_type='dataframe')

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

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
