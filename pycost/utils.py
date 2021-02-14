
import pandas as pd
#from panel.widgets.input import PasswordInput
def get_fys(df):
    fys = df.columns.str.replace("FY", "").str.strip().str.isdigit()
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