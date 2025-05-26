"""
Learning curve calculation and analysis functions.

This module provides functions for learning curve calculations, including:
- Midpoint calculations
- Learning curve effects
- Data preparation for learning curve analysis
"""

import numpy as np
import pandas as pd

def asher_midpoint(first_unit, last_unit, slope):
    """
    Calculate the midpoint of a learning curve using Asher's method.
    
    Parameters:
    -----------
    first_unit : float
        Cost/time of the first unit
    last_unit : float
        Cost/time of the last unit
    slope : float
        Learning curve slope (must be between 0 and 1)
        
    Returns:
    --------
    float
        Midpoint value
        
    Raises:
    -------
    ValueError
        If slope is not between 0 and 1
    """
    if slope <= 0 or slope >= 1:
        raise ValueError("Slope must be between 0 and 1 for learning curves.")
    
    b = np.log(slope) / np.log(2)  # Learning exponent
    num = last_unit**(b + 1) - first_unit**(b + 1)
    denom = (b + 1) * (last_unit**b - first_unit**b)
    return num / denom


def lc_midpoint(first_unit, last_unit, LC=1):
    """
    Calculate learning curve midpoint using approximation method.
    
    Parameters:
    -----------
    first_unit : float
        Cost/time of the first unit
    last_unit : float
        Cost/time of the last unit
    LC : float, default=1
        Learning curve slope
        
    Returns:
    --------
    float
        Midpoint value
    """
    try:
        b = np.log(LC)
    except:
        b = 0
    if b == 0:
        return (first_unit+last_unit + 2 * (first_unit*last_unit)**0.5)/4
    else:
        return (first_unit+last_unit + 2 * (first_unit*last_unit)**0.5)/4

def learn_curve(T1, LC, RC, Qty, Rate):
    """
    Calculate total cost using learning curve and rate curve effects.
    
    Parameters:
    -----------
    T1 : float
        First unit cost/time
    LC : float
        Learning curve slope (decimal form, e.g., 0.9 for 90%)
    RC : float
        Rate curve slope (decimal form)
    Qty : float
        Total quantity to produce
    Rate : float
        Production rate
        
    Returns:
    --------
    float
        Total cost with learning and rate effects
    """
    return T1 * Qty**(np.log(LC)/np.log(2)) * Rate ** (np.log(RC) / np.log(2))

def lc(T1, LC, RC, Qty, Rate):
    """
    Alias for learn_curve function.
    
    See learn_curve for documentation.
    """
    return T1 * Qty**(np.log(LC)/np.log(2)) * Rate ** (np.log(RC) / np.log(2))

def lc_prep(df, cols, val="value", lc_slope=1):
    """
    Calculate first unit, last unit, and midpoint for each group.
    
    Prepares data for learning curve analysis by grouping and calculating
    the necessary values for each group.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    cols : list
        Columns to group by
    val : str, default="value"
        Column containing the quantity values
    lc_slope : float, default=1
        Learning curve slope to use for midpoint calculation
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added columns: First, Last, midpoint, share_qty
    """
    # Handle pandas version differences
    if pd.__version__ >= '1.0':
        lc = df.groupby(cols)[val].agg(share_qty='sum').reset_index()
    else:
        lc = df.groupby(cols)[val].agg(sum).reset_index()
        lc = lc.rename(columns={val: "share_qty"})
        
    # Calculate cumulative quantities
    lc["Last"] = lc.groupby(cols[:-1])["share_qty"].cumsum()
    lc["First"] = lc["Last"] - lc["share_qty"] + 1
    lc["midpoint"] = lc_midpoint(lc["First"], lc["Last"], lc_slope)
    
    # Keep only necessary columns and merge back to original dataframe
    lc = lc[cols + ['First', 'Last', 'midpoint', 'share_qty']]
    lc = pd.merge(df, lc, how='left', on=cols, sort=False)
    
    return lc
