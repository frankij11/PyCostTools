import pandas as pd
import numpy as np
import datetime
import logging
import sys
import traceback

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

__all__=["AutoPreProcess", "DateTransform", "FeatureCheck", "ImputeNA", "MakeFormula", "Clean", "setup_logging"]

# Setup logging configuration
def setup_logging(log_level=logging.INFO, log_file=None):
    """Configure logging for the module.
    
    Args:
        log_level: The logging level (default: INFO)
        log_file: Optional file path for logging (default: None - console only)
    
    Returns:
        Logger instance for this module
    """
    logger = logging.getLogger('pycost.process')
    logger.setLevel(log_level)
    logger.propagate = False
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Create formatters and handlers - add filename and line number to format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Initialize the logger with default settings
logger = setup_logging()

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
    
    def get_feature_names_out(self, input_features=None):
        '''
        Get output feature names for transformation.
        '''
        return self.pipeline.get_feature_names_out(input_features)

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
            
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if input_features is None:
            input_features = self.date_columns
        
        output_features = []
        for col in input_features:
            if col in self.date_columns:
                if self.cont_year: output_features.append(f"{col}_cont_year")
                if self.year: output_features.append(f"{col}_year")
                if self.month: output_features.append(f"{col}_month")
                if self.day: output_features.append(f"{col}_day")
                if self.weekday: output_features.append(f"{col}_weekday")
                if self.season: output_features.append(f"{col}_season")
                if not self.drop: output_features.append(col)
            else:
                output_features.append(col)
        
        return np.array(output_features)

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


        if self.add_features and len(add_cols) > 0:
            print("add_cols", add_cols)
            #logger.debug(f"Adding {len(add_cols)} features to dataset")
            for col in add_cols:
                X[col] = np.nan
        else:
            logger.debug(f"failed feature check",f"{len(add_cols)} featuers not present in dataset", end="/n")
            #raise ValueError(f"{add_cols} not present in dataset")
        if self.coerce_type:
            for col in self.columns:
                X[col] = X[col].astype(self.sample_data[col].dtype, errors="ignore")

        return X
        
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        # This transformer ensures all original columns are present
        return np.array(self.columns)

class ImputeNA(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder."""

    def __init__(self,numeric_imputer=SimpleImputer(strategy='median'), categorical_imputer=SimpleImputer(strategy='most_frequent'),**kwargs):
        logger.debug("Initializing ImputeNA")
        self.numeric_imputer = numeric_imputer
        self.categorical_imputer = categorical_imputer
        
    def fit(self, X:pd.DataFrame, y=None):
        logger.debug("Fitting ImputeNA")
        try:
            X = X.copy()
            self.columns = X.columns.tolist()
            self.num_cols = X.select_dtypes(include=np.number).columns.tolist()
            self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            logger.debug(f"Data has {len(self.num_cols)} numeric columns, {len(self.cat_cols)} categorical columns")

            if len(self.num_cols) > 0: 
                logger.debug(f"Fitting numeric imputer on {len(self.num_cols)} columns")
                self.numeric_imputer = self.numeric_imputer.fit(X[self.num_cols])
                
            if len(self.cat_cols) > 0: 
                logger.debug(f"Fitting categorical imputer on {len(self.cat_cols)} columns")
                self.categorical_imputer = self.categorical_imputer.fit(X[self.cat_cols])
            
            # Convert set to list to avoid DataFrame indexing with set issues
            self.rest_cols = list(set(self.columns) - set(self.num_cols) - set(self.cat_cols))
            logger.debug(f"Identified {len(self.rest_cols)} remaining columns")

            return self
        except Exception as e:
            logger.error(f"Error during ImputeNA fitting: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            raise

    def transform(self, X:pd.DataFrame):
        logger.debug("Transforming data with ImputeNA")
        try:
            X = X.copy()
            cols = X.columns.tolist()
            add_cols = set(self.columns) - set(cols)
            
            if add_cols:
                logger.debug(f"Adding {len(add_cols)} missing columns with NaN values")
                X[list(add_cols)] = np.nan

            # Process numeric columns
            if len(self.num_cols) > 0: 
                logger.debug(f"Imputing {len(self.num_cols)} numeric columns")
                nums = self.numeric_imputer.transform(X[self.num_cols])
                nums = pd.DataFrame(nums, columns=self.num_cols)
            else:
                nums = pd.DataFrame()

            # Process categorical columns
            if len(self.cat_cols) > 0: 
                logger.debug(f"Imputing {len(self.cat_cols)} categorical columns")
                cats = self.categorical_imputer.transform(X[self.cat_cols])
                cats = pd.DataFrame(cats, columns=self.cat_cols)
            else:
                cats = pd.DataFrame()

            # Process remaining columns (using list, not set, for indexing)
            if self.rest_cols:
                logger.debug(f"Processing {len(self.rest_cols)} remaining columns")
                rest = X[self.rest_cols]
            else:
                rest = pd.DataFrame()

            # Combine all columns back together
            logger.debug("Combining imputed data")
            new_X = pd.concat([nums, cats, rest], axis=1)[self.columns]
            logger.debug(f"ImputeNA transform complete, output shape: {new_X.shape}")

            return new_X
        except Exception as e:
            logger.error(f"Error during ImputeNA transform: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            raise
            
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        # ImputeNA preserves all original feature names
        return np.array(self.columns)

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
            
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if input_features is not None:
            return np.array(input_features)
        return np.array(self.columns)

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
    
    def __init__(self,formula='`', handle_na=ImputeNA(),return_type='dataframe',keep_cols="all", return_X=True,return_y=False, wildcard="`"):
        logger.debug(f"Initializing MakeFormula with formula: {formula}")
        self.formula = formula
        self.wildcard = wildcard
        
        self.handle_na = handle_na
        if self.handle_na==False: 
            logger.debug("Setting handle_na to None (will drop NA values)")
            self.handle_na = None
        if self.handle_na==True: 
            logger.debug("Creating default ImputeNA instance")
            self.handle_na = ImputeNA()

        self.return_type = return_type
        self.keep_cols = keep_cols
        self.return_X = return_X
        self.return_y = return_y
        logger.debug(f"MakeFormula config: return_X={return_X}, return_y={return_y}, return_type={return_type}")

    
    def __getstate__(self):
        '''Pickle Instructions'''
        logger.debug("Pickling MakeFormula instance")
        self.y = None
        self.X = None
        return self.__dict__
    
    def __setstate__(self, d):
        '''Unpickle Instructions'''
        logger.debug("Unpickling MakeFormula instance")
        self.__dict__ = d
        try:
            if hasattr(self, 'df'):
                if not self.df.empty():
                    logger.debug("Refitting with stored dataframe")
                    self = self.fit(self.df)
        except Exception as e:
            logger.error(f"Error during MakeFormula unpickling: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            
    
    def fit(self, X, y=None):
        logger.debug(f"Fitting MakeFormula with input shape: {X.shape}")
        try:
            # handle NAs
            X = X.copy()
            if not self.handle_na is None:
                logger.debug("Fitting NA handler") 
                self.handle_na.fit(X)

            # replace wildcard in formula
            logger.debug(f"Parsing wildcard in formula: {self.formula}")
            self.formula = self.parse_formula_wildcard(self.formula, X, wildcard=self.wildcard)
            logger.debug(f"Parsed formula: {self.formula}")
                    
            # parse formula
            self.split_formula = self.formula.split("~")
            
            # keep copy of data
            self.df = X

            # find design matrix formula
            logger.debug("Creating patsy design matrices")
            if len(self.split_formula) > 1:
                logger.debug("Formula contains both target and features")
                self.y, self.X = patsy.dmatrices(self.formula, self.df)
                self.y = self.y.design_info
                self.X = self.X.design_info
            else:
                logger.debug("Formula contains only features")
                self.X = patsy.dmatrix(self.formula, self.df)
                self.X = self.X.design_info
            
            logger.debug("MakeFormula fit complete")
            return self
            
        except Exception as e:
            logger.error(f"Error during MakeFormula fit: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            raise

    @staticmethod
    def parse_formula_wildcard(formula,df, wildcard="`"):
        # find wildcard in formula
        logger.debug(f"Parsing wildcard in formula: {formula}")
        
        if "~" in formula:
            logger.debug("Getting target columns from formula")
            target = MakeFormula.get_formula_cols(formula, df, target_val=True, feature_vals=False)
        else:
            target = []
            
        if wildcard in formula:
            logger.debug(f"Wildcard '{wildcard}' found in formula")
            # add check to see if variable can be added
            all_cols = []
            not_added = []
            for col in [f"{col}" for col in df.columns if not col in target]:
                try:
                    tmp = patsy.dmatrix(col, df)
                    all_cols.append(col)
                except Exception as e:
                    logger.debug(f"Could not add column '{col}' to formula: {str(e)}")
                    not_added.append(col)
                    
            all_cols = " + ".join(all_cols)
            formula = formula.replace(wildcard, all_cols)
            logger.debug(f"Expanded formula: {formula}")
        else:
            logger.debug("No wildcard found in formula")
        
        return formula

    @staticmethod
    def get_formula_cols(formula, df, target_val=False, feature_vals=False):
        logger.debug(f"Getting columns from formula: target={target_val}, features={feature_vals}")
        if target_val:
            formula = formula.split("~")[0]
        if feature_vals:
            formula = formula.split("~")[1]
            
        # test just the first 2 datapoints so it runs quicker
        df = df.sample(min(2, len(df)))
        cols = []
        
        for col in df.columns:
            try:
                if target_val | feature_vals:
                    tmp_mod = patsy.dmatrix(formula, df.drop(col, axis=1))
                else:
                    tmp_mod = patsy.dmatrices(formula, df.drop(col, axis=1))
            except:
                cols.append(col)
                
        logger.debug(f"Found {len(cols)} columns: {cols}")
        return cols


    def transform(self, X):
        logger.debug(f"Transforming data with MakeFormula, input shape: {X.shape}")
        try:
            X = X.copy()
            if not self.handle_na is None: 
                logger.debug("Handling NA values before transform")
                X = self.handle_na.transform(X)

            logger.debug(f"Building design matrices with return_X={self.return_X}, return_y={self.return_y}")
            if self.return_X & self.return_y:
                logger.debug("Returning both X and y transformed")
                X_transform = patsy.build_design_matrices([self.X], X, return_type=self.return_type)[0]
                y_transform = patsy.build_design_matrices([self.y], X, return_type=self.return_type)[0]
                ans = (y_transform, X_transform)
            
            elif self.return_X:
                logger.debug("Returning only X transformed")
                X_transform = patsy.build_design_matrices([self.X], X, return_type=self.return_type)[0]
                ans = X_transform
            
            elif self.return_y:
                logger.debug("Returning only y transformed")
                y_transform = patsy.build_design_matrices([self.y], X, return_type=self.return_type)[0]
                ans = y_transform
            
            else:
                logger.error("Invalid configuration: at least one of return_X or return_y must be True")
                raise ValueError(self, "Need to choose an return X or return Y")

            if hasattr(ans, 'shape'):
                logger.debug(f"Transform complete, output shape: {ans.shape}")
            else:
                logger.debug("Transform complete, returning tuple of matrices")
                
            return ans
            
        except Exception as e:
            logger.error(f"Error during MakeFormula transform: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            raise

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if hasattr(self, 'X'):
            if self.return_type == 'dataframe':
                # Get column names from design_info
                if self.return_X:
                    return np.array(self.X.column_names)
                elif self.return_y:
                    return np.array(self.y.column_names)
            elif hasattr(self.X, 'column_names'):
                return np.array(self.X.column_names)
            
        # Fallback if we don't have the info we need
        if input_features is not None:
            return np.array(input_features)
        return np.array([])

class LC_Lot_Midpoint(BaseEstimator, TransformerMixin):
    '''
    routine to automatically calculate First Unit, Last Unit, MidpointQty
    '''

    def __init__(self, meta_columns=[], lot_order_columns=['FiscalYear'], quantity_column='value', priors_column = None, lc_slope=1, lot_qty_col ='lot_qty', lot_midpoint_col='lot_midpoint'):
        logger.debug(f"Initializing LC_Lot_Midpoint with quantity_column={quantity_column}, lc_slope={lc_slope}")
        self.meta_columns = meta_columns
        self.lot_order_columns = lot_order_columns
        self.quantity_column = quantity_column
        self.priors_column = priors_column
        self.lc_slope = lc_slope
        self.lot_qty_col = lot_qty_col
        self.lot_midpoint_col = lot_midpoint_col
        logger.debug(f"LC_Lot_Midpoint config: meta_columns={meta_columns}, lot_order_columns={lot_order_columns}")

    def fit(self, X, y=None):
        logger.debug(f"Fitting LC_Lot_Midpoint (no-op)")
        # Nothing to do...maybe store priors?
        return self
    

    def transform(self, X):
        logger.info(f"Transforming data with LC_Lot_Midpoint")
        try:
            df = X.copy()
            logger.debug(f"Input shape: {df.shape}")
            
            # Perform learning curve preparation
            df = self.lc_prep(
                df = df,
                lot_order_columns = [*self.meta_columns, *self.lot_order_columns], 
                quantity_column=self.quantity_column,
                priors_column= self.priors_column,
                lc_slope = self.lc_slope,
                lot_qty_col=self.lot_qty_col,
                lot_midpoint_col=self.lot_midpoint_col ) 
                
            logger.debug(f"LC preparation complete, output shape: {df.shape}")
            logger.debug(f"New columns added: First, Last, {self.lot_midpoint_col}, {self.lot_qty_col}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error during LC_Lot_Midpoint transform: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            raise

    @staticmethod
    def lc_prep(df, lot_order_columns, quantity_column="value", priors_column=None, lc_slope=1, lot_qty_col='lot_qty', lot_midpoint_col='lot_midpoint'):
        logger.debug(f"Preparing learning curve data with columns={lot_order_columns}, quantity_column={quantity_column}, lc_slope={lc_slope}")
        
        try:
            # Create lot quantity data
            if pd.__version__ >= '1.0':
                logger.debug("Using pandas 1.0+ aggregation method")
                lc = df.groupby(lot_order_columns)[quantity_column].agg(share_qty = 'sum').reset_index()
                lc = lc.rename(columns = {'share_qty': lot_qty_col})
            else:
                logger.debug("Using pandas pre-1.0 aggregation method")
                lc = df.groupby(lot_order_columns)[quantity_column].agg(sum).reset_index()
                lc = lc.rename(columns={quantity_column:lot_qty_col})
                
            # Calculate cumulative quantities
            if len(lot_order_columns) > 1:
                logger.debug(f"Calculating 'Last' using groupby on {lot_order_columns[:-1]}")
                lc["Last"] = lc.groupby(lot_order_columns[:-1])[lot_qty_col].cumsum()
            else:
                logger.debug("Calculating 'Last' from simple cumsum")
                lc['Last'] = lc[lot_qty_col].cumsum()
                
            # Calculate first units and add empty midpoint column
            lc["First"] = lc["Last"] - lc[lot_qty_col] + 1
            lc[lot_midpoint_col] = np.nan  # wait to calculate until we have a priors column
            
            # Keep only necessary columns
            lc = lc[lot_order_columns + ['First', 'Last', lot_midpoint_col, lot_qty_col]]
            
            # Merge back with original data
            logger.debug("Merging processed data back with original DataFrame")
            lc = pd.merge(df, lc, how='left', on=lot_order_columns, sort=False, suffixes=("_orig", ""))
            
            # Apply priors if needed
            if priors_column is None:
                logger.debug("No priors specified, using 0")
                priors = 0
            else:
                logger.debug(f"Using priors from column: {priors_column}")
                priors = lc[priors_column]
                
            # Adjust for priors and calculate midpoints
            lc['Last'] = lc['Last'] + priors
            lc['First'] = lc['First'] + priors
            
            logger.debug("Calculating lot midpoints")
            lc[lot_midpoint_col] = LC_Lot_Midpoint.lc_midpoint(lc["First"], lc["Last"], lc_slope)
            
            logger.debug(f"LC preparation complete, output shape: {lc.shape}")
            return lc
            
        except Exception as e:
            logger.error(f"Error in lc_prep: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            raise

    @staticmethod
    def lc_midpoint(first, last, lc_slope):
        logger.debug(f"Calculating midpoints with lc_slope={lc_slope}")
        try:
            b = np.log(lc_slope)
            if b == 0:
                logger.debug("Using arithmetic midpoint calculation (slope = 1)")
                return (first+last + 2 *(first*last)**.5)/4
            else:
                logger.debug("Using learning curve midpoint calculation")
                midpoint = ((1 / (last - first + 1)) * ((((last + 0.5) ** (1 + b)) - ((first - 0.5) ** (1 + b))) / (1 + b))) ** (1 / b)
                return midpoint
                
        except Exception as e:
            logger.error(f"Error calculating midpoints: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            raise
            
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if input_features is None:
            # We don't know the exact input features without seeing data
            return np.array([])
            
        # This transformer adds several new columns
        output_features = list(input_features)
        
        # Add the new columns this transformer creates
        new_columns = ['First', 'Last', self.lot_midpoint_col, self.lot_qty_col]
        
        # Add new columns that aren't already in the input
        for col in new_columns:
            if col not in output_features:
                output_features.append(col)
                
        return np.array(output_features)

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
        
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if input_features is None:
            # If no input features provided, we can't determine output features
            return np.array([])
        
        # This transformer replaces the values of categorical columns but keeps the same column names
        return np.array(input_features)

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
        
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if input_features is None:
            # If no input features provided, we can't determine output features
            return np.array([])
        
        # This transformer maintains the original column names
        return np.array(input_features)

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
        
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        # This is a complex transformer that creates new columns from categorical variables
        output_features = []
        
        # Add numerical features that aren't in drop_cols
        for var in self.num_vars:
            if var not in self.drop_cols:
                output_features.append(var)
        
        # Add one-hot encoded categorical features
        for var in self.cat_vars:
            for cat in self.cats_[var]:
                col_name = var + "_" + cat
                output_features.append(col_name)
        
        return np.array(output_features)
