import pandas as pd
def get_fys(df):
    fys = df.columns.str.replace("FY", "").str.strip().str.isdigit()
    return fys

def stack_fys(df, fy_name="FY"):
    fys = get_fys(df)
    df_stacked = pd.melt(df, id_vars=df.columns[~fys], value_vars=df.columns[fys], var_name=fy_name)
    return df_stacked