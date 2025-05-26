"""
Inflation calculation and adjustment functions.

This module provides functions for inflation adjustments using JIC (Joint Inflation Calculator)
data, supporting both Base Year (BY) and Then Year (TY) conversions.
"""

import numpy as np
import pandas as pd
import os
from pycost.utils import stack_fys

# Try to import openpyxl, but provide fallbacks if not available
try:
    import openpyxl
    _has_openpyxl = True
except ImportError:
    _has_openpyxl = False

def make_jic(fName=None):
    """
    Create a Joint Inflation Calculator (JIC) dataset from an Excel file.
    
    Parameters:
    -----------
    fName : str, optional
        Path to the JIC Excel file. If None, uses the default packaged file.
        
    Returns:
    --------
    pandas.DataFrame
        Processed JIC data with raw, combo factors, and weighted values
        
    Notes:
    ------
    Requires openpyxl to be installed.
    """
    if not _has_openpyxl:
        print("Cannot make JIC: openpyxl not installed")
        return pd.DataFrame()
        
    if fName is None:
        location = os.path.dirname(os.path.realpath(__file__))
        fName = os.path.join(location, 'data', 'JIC_PB21_2_6_2020_Final.xlsm')

    xl = openpyxl.load_workbook(fName, read_only=True, data_only=True)

    def getNrows(sh='1970=1 Infl Index'):
        """Get number of rows with data in a sheet."""
        i = 0
        nRows = None
        for r in xl[sh]['A7':'A1000']:
            if r[0].value is None:
                nRows = i
                break
            i = i + 1
        return nRows
    
    nRows = getNrows()
    jic_raw = pd.read_excel(fName,
                            sheet_name="1970=1 Infl Index",
                            header=0, usecols="A:BD", skiprows=5,
                            engine="openpyxl",
                            nrows=getNrows("1970=1 Infl Index"))
    
    jic_combof = pd.read_excel(fName,
                               sheet_name="CombOutFac(COF)",
                               header=0, usecols="A:BD", skiprows=5,
                               engine="openpyxl",
                               nrows=getNrows("CombOutFac(COF)"))
    
    meta = (
        pd.read_excel(fName, sheet_name="Titles", header=0, usecols="B:G", skiprows=9, engine="openpyxl")
          .rename(columns={"Idx": "service", "Perm": "id", "ShortTitle": "short_title", "Year": "fy", "Long Title": "long_title", "OSD": "osd", "TreasCode": "treas_code"})
          .assign(source=os.path.basename(fName), Date=xl.properties.modified)
    )
    
    def forecast_years(s, forecast=True):
        """
        Forecast missing values in a series.
        
        Parameters:
        -----------
        s : pandas.Series
            Series of values
        forecast : bool, default=True
            If True, use growth rate for forecasting. If False, use last value.
            
        Returns:
        --------
        list
            Complete series with forecasted values
        """
        vals = []
        for i in range(len(s)):
            if not np.isnan(s.iloc[i]):
                vals.append(s.iloc[i])
            else:
                if forecast:
                    vals.append(vals[i-1] / vals[i-2] * vals[i-1])
                else:
                    vals.append(vals[i-1])
        return vals

    # Add forecasted years up to 2100
    addYears = range(jic_raw.Year.iloc[-1], 2100) 
    jic_raw = pd.concat([jic_raw, pd.DataFrame({'Year': addYears})], ignore_index=True)
    jic_raw.iloc[:, 1:] = jic_raw.iloc[:, 1:].apply(forecast_years, axis=0)
    jic_raw = pd.melt(jic_raw, id_vars='Year', value_name='raw', var_name='id')

    jic_combof = pd.concat([jic_combof, pd.DataFrame({'Year': addYears})], ignore_index=True)
    jic_combof.iloc[:, 1:] = jic_combof.iloc[:, 1:].apply(forecast_years, forecast=False, axis=0) 
    jic_combof = pd.melt(jic_combof, id_vars='Year', value_name='combof', var_name='id')

    # Merge raw indices with combo factors
    jic = pd.merge(jic_raw, jic_combof, how='left', on=["Year", "id"])
    jic = jic.assign(wtd=jic.raw * jic.combof)

    # Merge metadata
    jic = meta.merge(jic, how='right', on="id").drop_duplicates()
    jic.index = range(jic.shape[0])
    jic = jic.rename(columns={"Year": "fy"})#.assign(index=jic.short_title)

    return jic

def change_BY(BY=2020, inflation_table=None):
    """
    Change the base year for inflation indices.
    
    Parameters:
    -----------
    BY : int, default=2020
        New base year
    inflation_table : pandas.DataFrame, optional
        JIC data. If None, creates new data using make_jic()
        
    Returns:
    --------
    pandas.DataFrame
        JIC data with updated base year
    """
    if inflation_table is None: 
        inflation_table = make_jic()
    
    by = inflation_table[['id', 'fy', 'raw']].query(f"fy == {BY}").rename(columns={"raw": "by_raw"})
    
    # Merge and update values
    new_inflation_table = pd.merge(inflation_table, by[['id', 'by_raw']], how='left', on='id')
    new_inflation_table = new_inflation_table.assign(
        raw=new_inflation_table.raw / new_inflation_table.by_raw, 
        wtd=new_inflation_table.raw / new_inflation_table.by_raw * new_inflation_table.combof
    ).drop(columns=["by_raw"])

    return new_inflation_table



# Load JIC data at module import time
location = os.path.dirname(os.path.realpath(__file__))
jic_file = os.path.join(location, 'data', 'JIC_PB21_2_6_2020_Final.xlsm')

jic = make_jic(jic_file)

class Inflation:
    """
    Inflation class to convert costs between years and types.
    
    This class provides a convenient interface for converting costs between
    different year types (Calendar Year and Then Year) using inflation indices.
    
    Parameters
    ----------
    inflation_table : pandas.DataFrame, optional
        Custom inflation table. If None, uses the package default.
    """
    def __init__(self, inflation_table=None, base_year=2020):
        if inflation_table is None:
            self.inflation_table = jic
        else:
            self.inflation_table = inflation_table
        self.base_year = base_year
        self.inflation_table = change_BY(BY=base_year, inflation_table=self.inflation_table)

    def CYtoCY(self, from_year, to_year=2023, index="GII", value=1, return_df=False):
        """
        Convert costs from one calendar year to another calendar year.
        
        Parameters
        ----------
        from_year : int or pandas.Series
            Source calendar year
        to_year : int, default=2023
            Target calendar year
        index : str, default="GII"
            Inflation index to use
        value : float, default=1
            Cost value to convert
        return_df : bool, default=False
            If True, returns the full DataFrame with calculations
            
        Returns
        -------
        float or pandas.Series or pandas.DataFrame
            Converted cost value(s) or full DataFrame
        """
        if isinstance(from_year, pd.Series):
            df = (pd.DataFrame(dict(from_year=from_year))
                .assign(to_year=to_year, index=index, value=value)
                .merge(self.inflation_table.assign(from_year=lambda x: x.fy, index=lambda x: x.short_title, div_by=lambda x: x.raw)[['from_year', 'index', 'div_by']], on=['from_year', 'index'], how='left')
                .merge(self.inflation_table.assign(to_year=lambda x: x.fy, index=lambda x: x.short_title, mult_by=lambda x: x.raw)[['to_year', 'index', 'mult_by']], on=['to_year', 'index'], how='left')
                .assign(norm_value=lambda x: x.value / x.div_by * x.mult_by)
            )
            if return_df:
                return df
            else:
                return df.norm_value.values
        else:
            div_by = self.inflation_table.query('fy==@from_year and short_title==@index')['raw'].values[0]
            mult_by = self.inflation_table.query('fy==@to_year and short_title==@index')['raw'].values[0]
            return value / div_by * mult_by

    def CYtoTY(self, from_year, to_year=2023, index="GII", value=1, return_df=False):
        """
        Convert costs from calendar year to then year.
        
        Parameters
        ----------
        from_year : int or pandas.Series
            Source calendar year
        to_year : int, default=2023
            Target then year
        index : str, default="GII"
            Inflation index to use
        value : float, default=1
            Cost value to convert
        return_df : bool, default=False
            If True, returns the full DataFrame with calculations
            
        Returns
        -------
        float or pandas.Series or pandas.DataFrame
            Converted cost value(s) or full DataFrame
        """
        if isinstance(from_year, pd.Series):
            df = (pd.DataFrame(dict(from_year=from_year))
                .assign(to_year=to_year, index=index, value=value)
                .merge(self.inflation_table.assign(from_year=lambda x: x.fy, index=lambda x: x.short_title, div_by=lambda x: x.raw)[['from_year', 'index', 'div_by']], on=['from_year', 'index'], how='left')
                .merge(self.inflation_table.assign(to_year=lambda x: x.fy, index=lambda x: x.short_title, mult_by=lambda x: x.wtd)[['to_year', 'index', 'mult_by']], on=['to_year', 'index'], how='left')
                .assign(norm_value=lambda x: x.value / x.div_by * x.mult_by)
            )
            if return_df:
                return df
            else:
                return df.norm_value.values
        else:
            div_by = self.inflation_table.query('fy==@from_year and short_title==@index')['raw'].values[0]
            mult_by = self.inflation_table.query('fy==@to_year and short_title==@index')['wtd'].values[0]
            return value / div_by * mult_by
    
    def TYtoCY(self, from_year, to_year=2023, index="GII", value=1, return_df=False):
        """
        Convert costs from then year to calendar year.
        
        Parameters
        ----------
        from_year : int or pandas.Series
            Source then year
        to_year : int, default=2023
            Target calendar year
        index : str, default="GII"
            Inflation index to use
        value : float, default=1
            Cost value to convert
        return_df : bool, default=False
            If True, returns the full DataFrame with calculations
            
        Returns
        -------
        float or pandas.Series or pandas.DataFrame
            Converted cost value(s) or full DataFrame
        """
        if isinstance(from_year, pd.Series):
            df = (pd.DataFrame(dict(from_year=from_year))
                .assign(to_year=to_year, index=index, value=value)
                .merge(self.inflation_table.assign(from_year=lambda x: x.fy, index=lambda x: x.short_title, div_by=lambda x: x.wtd)[['from_year', 'index', 'div_by']], on=['from_year', 'index'], how='left')
                .merge(self.inflation_table.assign(to_year=lambda x: x.fy, index=lambda x: x.short_title, mult_by=lambda x: x.raw)[['to_year', 'index', 'mult_by']], on=['to_year', 'index'], how='left')
                .assign(norm_value=lambda x: x.value / x.div_by * x.mult_by)
            )
            if return_df:
                return df
            else:
                return df.norm_value.values
        else:
            div_by = self.inflation_table.query('fy==@from_year and short_title==@index')['wtd'].values[0]
            mult_by = self.inflation_table.query('fy==@to_year and short_title==@index')['raw'].values[0]
            return value / div_by * mult_by
    
    def TYtoTY(self, from_year, to_year=2023, index="GII", value=1, return_df=False):
        """
        Convert costs from one then year to another then year.
        
        Parameters
        ----------
        from_year : int or pandas.Series
            Source then year
        to_year : int, default=2023
            Target then year
        index : str, default="GII"
            Inflation index to use
        value : float, default=1
            Cost value to convert
        return_df : bool, default=False
            If True, returns the full DataFrame with calculations
            
        Returns
        -------
        float or pandas.Series or pandas.DataFrame
            Converted cost value(s) or full DataFrame
        """
        if isinstance(from_year, pd.Series):
            df = (pd.DataFrame(dict(from_year=from_year))
                .assign(to_year=to_year, index=index, value=value)
                .merge(self.inflation_table.assign(from_year=lambda x: x.fy, index=lambda x: x.short_title, div_by=lambda x: x.wtd)[['from_year', 'index', 'div_by']], on=['from_year', 'index'], how='left')
                .merge(self.inflation_table.assign(to_year=lambda x: x.fy, index=lambda x: x.short_title, mult_by=lambda x: x.wtd)[['to_year', 'index', 'mult_by']], on=['to_year', 'index'], how='left')
                .assign(norm_value=lambda x: x.value / x.div_by * x.mult_by)
            )
            if return_df:
                return df
            else:
                return df.norm_value.values
        else:
            div_by = self.inflation_table.query('fy==@from_year and short_title==@index')['wtd'].values[0]
            mult_by = self.inflation_table.query('fy==@to_year and short_title==@index')['wtd'].values[0]
            return value / div_by * mult_by
    
    def get_indices_labels(self):
        """
        Get all available inflation indices.
        
        Returns
        -------
        numpy.ndarray
            Array of unique category labels in the inflation table
        """
        return self.inflation_table.short_title.unique()
    
    def change_base_year(self, base_year):
        """
        Change the base year for the inflation table.
        """
        df = change_BY(BY=base_year, inflation_table=self.inflation_table)
        self.inflation_table = df
        return df
    
    
    def to_wide(self, stack=True, df=None):
        """
        Convert the inflation table to wide format.
        
        Parameters
        ----------
        stack : bool, default=True
            If True, returns a single DataFrame with raw and wtd data stacked.
            If False, returns a tuple of (raw_df, wtd_df).
        df : pandas.DataFrame, optional
            Custom inflation table to convert. If None, uses self.inflation_table.
            
        Returns
        -------
        pandas.DataFrame or tuple
            Inflation data in wide format
        """
        if df is None:
            df = self.inflation_table
            
        pivot_cols = ['fy']
        value_cols = ['raw', 'wtd']
        cols = ['source','date','id','service','osd','long_title', 'short_title']
        infl_cols_lower = [c.lower() for c in df.columns]
        cols = [c for c in cols if c.lower() in infl_cols_lower]
        raw_df = (df.copy()
                  .rename(columns=lambda x: x.lower())
                  .assign(type='raw')
              .pivot_table(index=cols+['type'], columns=pivot_cols, values=value_cols[0])
              .reset_index()
              )
        wtd_df = (df.copy()
                  .rename(columns=lambda x: x.lower())
                  .assign(type='wtd')
              .pivot_table(index=cols+['type'], columns=pivot_cols, values=value_cols[1])
              .reset_index()
              )
        if stack:
            return pd.concat([raw_df, wtd_df], axis=1)
        else:
            return raw_df, wtd_df

    def to_excel(self, file_name, base_year=None, long=True, stack=True, period_start=1900, period_end=2100, indices=None):
        """
        Save the inflation table to an Excel file.
        
        Parameters
        ----------
        file_name : str
            Path to save the Excel file
        base_year : int, optional
            Base year for inflation calculations. If None, uses self.base_year
        long : bool, default=True
            If True, saves data in long format. If False, uses wide format
        stack : bool, default=True
            If long=False, determines whether to stack raw and wtd data in one sheet
            or separate them into different sheets
        period_start : int, default=1900
            Filter data to years >= period_start
        period_end : int, default=2100
            Filter data to years <= period_end
        indices : str or list of str, optional
            Filter data to specific inflation indices. If None, includes all indices
            
        Returns
        -------
        None
            Saves data to Excel file
        """
        # change base year
        if base_year is None:
            base_year = self.base_year
        df = change_BY(BY=base_year, inflation_table=self.inflation_table)

        if indices:
            if isinstance(indices, str):
                indices = [indices]
            df = df.query("short_title in @indices")
        if period_start>0:
            df = df.assign(fy=lambda x: pd.to_numeric(x.fy, errors='coerce')).query("fy >= @period_start")
        if period_end>0:
            df = df.assign(fy=lambda x: pd.to_numeric(x.fy, errors='coerce')).query("fy <= @period_end")
        
        if long:
            df.to_excel(file_name, sheet_name="inflation", index=False)
        else:
            # stack raw and wtd columns
            if stack:
                wide_df = self.to_wide(stack=stack, df=df)
                wide_df.to_excel(file_name, sheet_name="inflation", index=False)
            else:
                raw, wtd = self.to_wide(stack=stack, df=df)
                with pd.ExcelWriter(file_name) as xl:
                    raw.to_excel(xl, sheet_name="raw", index=False)
                    wtd.to_excel(xl, sheet_name="wtd", index=False)



if __name__ == "__main__":
    jic2 = make_jic(fName=r"C:\Users\kevin\OneDrive\Documents\Projects\CaSES\PyCostTools\pycost\data\JIC_PB21_2_6_2020_Final.xlsm")
    infl_2025 = Inflation(inflation_table=jic2, base_year=2025)#.rename(columns={"Indice":"category", "Year":"fy", "Raw":"raw", "weighted":"wtd"}))
    infl = Inflation()
    print(infl.to_wide())
    print(jic.columns)
    print(jic2.columns)

    print(jic2.short_title.unique())
    print(jic2.assign(fy=lambda x: pd.to_numeric(x.fy, errors='coerce')).query("short_title == 'APN' and fy >= 2025 and fy <= 2030"))
    print(jic.assign(fy=lambda x: pd.to_numeric(x.fy, errors='coerce')).query("short_title == 'APN' and fy >= 2025 and fy <= 2030"))

    print(infl_2025.CYtoCY(2020, 2021, "APN", 100), infl.CYtoCY(2020, 2021, "APN", 100))
    print(infl_2025.CYtoTY(2020, 2021, "APN", 100), infl.CYtoTY(2020, 2021, "APN", 100))
    
    print(infl.TYtoCY(2020, 2021, "APN", 100), infl_2025.TYtoCY(2020, 2021, "APN", 100))
    
    print(infl.TYtoTY(2020, 2021, "APN", 100), infl_2025.TYtoTY(2020, 2021, "APN", 100))
    print(infl_2025.inflation_table.query("fy==2025"))
    print(infl.inflation_table.query("fy==2025"))
    infl_2025.to_excel("infl_2025.xlsx")
    infl_2025.to_excel("infl_2025_wide.xlsx", long=False, stack=True)
    infl_2025.to_excel("infl_2025_wide_nonstack.xlsx", long=False, stack=False)
    infl_2025.to_excel("infl_2025_wide_nonstack_period.xlsx", long=False, stack=False, period_start=2025, period_end=2030)


    
