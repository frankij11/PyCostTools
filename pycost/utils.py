"""
Utility functions and extensions for PyCost

This module provides general utility functions and pandas DataFrame extensions
that are useful for cost analysis and data manipulation.
"""

import pandas as pd
import re
import sys

# Try to import patsy, but provide fallbacks if not available
try:
    import patsy
    _has_patsy = True
except ImportError:
    _has_patsy = False

@pd.api.extensions.register_dataframe_accessor("ct")
class CostTools:
    """
    Pandas DataFrame accessor extension for cost analysis.
    
    This class provides methods for selecting and manipulating DataFrame columns
    in ways that are useful for cost analysis, including dplyr-inspired syntax.
    
    Access via the 'ct' accessor on any DataFrame: df.ct.method()
    """
    
    def __init__(self, pandas_obj):
        """Initialize the accessor with a pandas DataFrame."""
        self.df = pandas_obj

    def formula(self, formula):
        """
        Create a design matrix using patsy formula syntax.
        
        Parameters:
        -----------
        formula : str
            A patsy formula string
            
        Returns:
        --------
        pandas.DataFrame
            The design matrix
        """
        if not _has_patsy:
            print("Cannot use formula: patsy not installed")
            return self.df
        return patsy.dmatrix(formula, self.df, return_type='dataframe')
    
    def select(self, columns:str):
        """
        Select columns using dplyr-inspired syntax.
        
        Parameters:
        -----------
        columns : str
            A string expression specifying columns to select or remove.
            Supports: everything(), starts_with(), contains(), ends_with()
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with selected columns
        """
        # Parse data
        def delete_duplicates(pos:list):
            """Remove duplicate entries from a list while preserving order."""
            res = []
            [res.append(x) for x in pos if x not in res]
            return res
            
        def remove_negatives(neg:list, orig_res:list):
            """Remove items in neg list from orig_res list."""
            return [col for col in orig_res if col not in neg]

        def get_params_str(s:str, func:str):
            """Extract parameters from a function call string."""
            expr = s.replace(func, "").replace("(","").replace(")", "").replace('"', "").replace("'","").strip().split(",")
            def str2bool(v):
                return v.lower() == "true"
            if len(expr) > 1: 
                expr[1] = str2bool(expr[1])
            return expr
            
        columns = columns.strip()
        if columns[0] != "-": 
            columns = "+" + columns
        
        # Parse for positive and negatives
        parse = re.split(r"(\+|\-)", columns)
        pos = []
        neg = []
        
        for i in range(len(parse)):
            try:
                col = parse[i+1].strip()
                if "everything" in col: 
                    col = self.df.columns.tolist()
                if "starts_with" in col: 
                    col = self.starts_with(*get_params_str(col, "starts_with")).columns.tolist()
                if "contains" in col: 
                    col = self.contains(*get_params_str(col, "contains")).columns.tolist()
                if "ends_with" in col: 
                    col = self.ends_with(*get_params_str(col, "ends_with")).columns.tolist()
                
                if parse[i] == "+": 
                    if isinstance(col, str): 
                        col = [col]
                    pos.extend(col)
                if parse[i] == "-": 
                    if isinstance(col, str): 
                        col = [col]
                    neg.extend(col)
            except:
                pass
                
        res = delete_duplicates(pos)
        cols = remove_negatives(neg, res)
        
        return self.df[cols]
    
    def contains(self, string:str, case=True):
        """
        Select columns that contain a specific string.
        
        Parameters:
        -----------
        string : str
            String pattern to match in column names
        case : bool, default=True
            Whether to do case-sensitive matching
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with matched columns
        """
        cols = []
        for col in self.df.columns:
            if case:
                if re.search(string, col): 
                    cols.append(col)
            else:
                if re.search(string.lower(), col.lower()): 
                    cols.append(col)

        return self.df[cols]
    
    def starts_with(self, string:str, case=True):
        """
        Select columns that start with a specific string.
        
        Parameters:
        -----------
        string : str
            String pattern to match at the beginning of column names
        case : bool, default=True
            Whether to do case-sensitive matching
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with matched columns
        """
        string = "^" + string
        cols = []
        for col in self.df.columns:
            if case:
                if bool(re.search(string, col)): 
                    cols.append(col)
            else:
                if bool(re.search(string.lower(), col.lower())): 
                    cols.append(col)
        return self.df[cols]
    
    def ends_with(self, string:str, case=True):
        """
        Select columns that end with a specific string.
        
        Parameters:
        -----------
        string : str
            String pattern to match at the end of column names
        case : bool, default=True
            Whether to do case-sensitive matching
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with matched columns
        """
        string = ".*" + string + "$"
        cols = []
        for col in self.df.columns:
            if case:
                if bool(re.match(string, col)): 
                    cols.append(col)
            else:
                if bool(re.match(string.lower(), col.lower())): 
                    cols.append(col)
        return self.df[cols]

    def get_fys(self, FY:str='FY|FiscalYear|Fiscal Year|Fiscal_Year'):
        """
        Identify fiscal year columns in the DataFrame.
        
        Parameters:
        -----------
        FY : str, default='FY|FiscalYear|Fiscal Year|Fiscal_Year'
            Regex pattern to identify fiscal year column prefixes
            
        Returns:
        --------
        pandas.Series
            Boolean series indicating which columns are fiscal years
        """
        return get_fys(self.df, FY)
    
    def stack_fys(self, fy_name="FY"):
        """
        Stack fiscal year columns into a long format.
        
        Parameters:
        -----------
        fy_name : str, default="FY"
            Name for the fiscal year column in the result
            
        Returns:
        --------
        pandas.DataFrame
            Long-format DataFrame with stacked fiscal years
        """
        return stack_fys(self.df, fy_name)


def get_fys(df, FY:str='FY|FiscalYear|Fiscal Year|Fiscal_Year'):
    """
    Identify fiscal year columns in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
    FY : str, default='FY|FiscalYear|Fiscal Year|Fiscal_Year'
        Regex pattern to identify fiscal year column prefixes
        
    Returns:
    --------
    pandas.Series
        Boolean series indicating which columns are fiscal years
    """
    fys = df.columns.str.replace(FY, "", regex=True).str.strip().str.isdigit()
    return fys

def stack_fys(df, fy_name="FY"):
    """
    Stack fiscal year columns into a long format.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to transform
    fy_name : str, default="FY"
        Name for the fiscal year column in the result
        
    Returns:
    --------
    pandas.DataFrame
        Long-format DataFrame with stacked fiscal years
    """
    fys = get_fys(df)
    df_stacked = pd.melt(df, id_vars=df.columns[~fys], value_vars=df.columns[fys], var_name=fy_name)
    return df_stacked


def get_imports():
    """
    Get a list of imported packages and their versions.
    
    Returns:
    --------
    list
        List of tuples containing (package_name, version)
    """
    g = dict(globals())
    results = []
    for var in g:
        if type(g[var]) == type(sys):
            try:
                if g[var].__package__ != "":
                    pkg = sys.modules[g[var].__package__]
                    results.append((pkg.__package__, pkg.__version__))
            except (AttributeError, KeyError):
                pass
    results = set(results)
    return list(results)

def make_requirements(fName="requirements.txt", imports=None, add_versions=False):
    """
    Generate a requirements.txt file from current imports.
    
    Parameters:
    -----------
    fName : str, default="requirements.txt"
        Output file name
    imports : list, optional
        List of imports to use; if None, uses get_imports()
    add_versions : bool, default=False
        Whether to include version constraints
    """
    # Get all imports
    if imports is None: 
        imports = get_imports()
    
    # Write to file
    with open(fName, "w") as f:
        for i in imports:
            if add_versions:
                f.write(f"{i[0]} >= {i[1]}\n")
            else:
                f.write(f"{i[0]}\n")