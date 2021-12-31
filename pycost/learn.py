import numpy as np
import pandas as pd



def lc_midpoint(f, l, LC=1):
    b = np.log(LC)
    if b == 0:
        return (f+l + 2 *(f*l)**.5)/4
    else:
        print("Asher's not implemented")
        return (f+l + 2 *(f*l)**.5)/4

def learn_curve(T1, LC, RC, Qty, Rate):
    return T1*Qty**(np.log(LC)/np.log(2)) * Rate ** (np.log(RC) / np.log(2))

def lc(T1, LC, RC, Qty, Rate):
    return T1*Qty**(np.log(LC)/np.log(2)) * Rate ** (np.log(RC) / np.log(2))

def lc_prep(df, cols, val= "value"):

    if pd.__version__ >= '1.0':
        lc =df.groupby(cols)[val].agg(share_qty = 'sum').reset_index()
    else:
        lc = df.groupby(cols)[val].agg(sum).reset_index()
        lc = lc.rename(columns={val:"share_qty"})
    lc["Last"] = lc.groupby(cols[:-1])["share_qty"].cumsum()
    lc["First"] = lc["Last"] - lc["share_qty"] + 1
    lc["midpoint"] = lc_midpoint(lc["First"], lc["Last"])
    lc = lc[cols+ ['First', 'Last', 'midpoint', 'share_qty']]
    lc = pd.merge(df,lc,how='left', on=cols,sort=False)
    return lc
