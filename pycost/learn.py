import numpy as np
import pandas as pd



def lc_midpoint(f, l, LC=1):
    b = np.log(LC)
    m =(f+l + 2 *(f+l)**.5)/4
    if b != 0:
        return m
    else:
        print("Asher's not implemented")
        return m

def learn_curve(T1, LC, RC, Qty, Rate):
    return T1*Qty**(np.log(LC)/np.log(2)) * Rate ** (np.log(RC) / np.log(2))

def lc(T1, LC, RC, Qty, Rate):
    return T1*Qty**(np.log(LC)/np.log(2)) * Rate ** (np.log(RC) / np.log(2))

def lc_prep(df, cols, val= "value"):

    lc =df.groupby(cols)[val].agg(share_qty = 'sum').reset_index()
    lc["Last"] = lc.groupby(cols[:-1])["share_qty"].cumsum()
    lc["First"] = lc["Last"] - lc["share_qty"] + 1
    lc["midpoint"] = lc_midpoint(df["First"], df["Last"])
    
    lc = pd.merge(df,lc,how='left', on=cols,sort=False)
    return lc
