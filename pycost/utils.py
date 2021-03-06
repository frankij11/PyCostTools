# %%
import pandas as pd
import re
import patsy

@pd.api.extensions.register_dataframe_accessor("ct")
class CostTools:
    '''
    Utility class to extend pandas dataframe
    '''
    copy=True
    def __init__(self, pandas_obj):
        self.df = pandas_obj

    def formula(self, formula):
        return patsy.dmatrix(formula, self.df, return_type='dataframe')
    
    def select(self,columns:str ):
        '''
        
        '''
        # parse data
        def delete_duplicates(pos:list):
            res = []
            [res.append(x) for x in pos if x not in res]
            
            return res
        def remove_negatives(neg:list, orig_res:list):
            res = []
            for col in orig_res:
                if col not in neg: res.append(col)
            
            return res

        def get_params_str(s:str, func:str):
            expr = col.replace(func, "").replace("(","").replace(")", "").replace('"', "").replace("'","").strip().split(",")
            def str2bool(v):
                return v.lower() == "true"
            if len(expr)>0: expr[1] = str2bool(expr[1])
            return expr
        columns = columns.strip()
        if columns[0] != "-": columns = "+" + columns
        
        # parse for positive and negatives does not take into account parantheses
        parse =re.split("(\\+|\\-)",columns)
        pos = []
        neg = []
        for i in range(len(parse)):
            try:
                col = parse[i+1].strip()
                if "everything" in col: col = self.df.columns
                if "starts_with" in col: col = self.starts_with(*get_params_str(col, "starts_with")).columns.tolist()
                if "contains" in col: col = self.contains(*get_params_str(col, "contains")).columns.tolist()
                if "ends_with" in col: col = self.ends_with(*get_params_str(col, "ends_with")).columns.tolist()
                if parse[i] == "+": 
                    if isinstance(col,str): col = [col]
                    for c in col:
                        pos.append(c)
                if parse[i] =="-": 
                    if isinstance(col,str): col = [col]
                    for c in col:
                        neg.append(c)

            except:
                pass
        res = delete_duplicates(pos)
        cols = remove_negatives(neg, res)
        
        return self.df[cols]
    
    def select_regex(self, search_string):
        pass

    def contains(self, string:str, case=True):
        '''
        columns that contains a certain string. uses regex
        '''
        cols = []
        for col in self.df.columns:
            if case:
                if re.search(string,col): cols.append(col)
            else:
                if re.search(string.lower() , col.lower()): cols.append(col)

        return self.df[cols]
    
    def starts_with(self, string:str, case=True):
        string = "^" + string
        cols =[]
        for col in self.df.columns:
            if case:
                if bool(re.search(string, col)): cols.append(col)
            else:
                if bool(re.search(string.lower(), col.lower())): cols.append(col)
        return self.df[cols]
    
    def ends_with(self, string:str, case=True):
        string = ".*" + string +"$"
        cols =[]
        for col in self.df.columns:
            if case:
                if bool(re.match(string, col)): cols.append(col)
            else:
                if bool(re.match(string.lower(), col.lower())): cols.append(col)

        return self.df[cols]


    def get_fys(self, FY:str='FY|FiscalYear|Fiscal Year|Fiscal_Year'):
        return get_fys(self.df, FY)
    
    def stack_fys(self, fy_name="FY"):
        return stack_fys(self.df,fy_name)


# %%
def get_fys(df, FY:str='FY|FiscalYear|Fiscal Year|Fiscal_Year'):

    fys = df.columns.str.replace(FY, "").str.strip().str.isdigit()
    return fys

def stack_fys(df, fy_name="FY"):
    fys = get_fys(df)
    df_stacked = pd.melt(df, id_vars=df.columns[~fys], value_vars=df.columns[fys], var_name=fy_name)
    return df_stacked


def get_imports():
    import sys
    g = dict(globals())
    results = []
    for var in g:
        if type(g[var]) == type(sys):
            try:
                if g[var].__package__ != "":
                    pkg = sys.modules[g[var].__package__]
                    results.append((pkg.__package__, pkg.__version__))
                #results.append((g[var].__name__, g[var].__package__))
            except:
                pass
    results = set(results)
    return list(results)

def make_requirements(fName="requirements.txt",imports=None,add_versions=False):
    # Run all modules first
    # get all imports
    if imports is None: imports = get_imports()
    
    # dump to fName
    f = open(fName, "w+")
    for i in imports:
        if add_versions:
            f.write(i[0] + " >= " + i[1] + "\n")
            
        else:
            f.write(i[0])
    f.close()
    pass