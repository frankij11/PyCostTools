import numpy as np
import pandas as pd
import os
from .utils import *
import openpyxl

def make_jic(fName=None):
    if fName is None:
        location = os.path.dirname(os.path.realpath(__file__))
        fName = os.path.join(location, 'data', 'JIC_PB21_2_6_2020_Final.xlsm')

    xl = openpyxl.load_workbook(fName, read_only=True, data_only=True)

    def getNrows(sh = '1970=1 Infl Index'):
        i = 0
        nRows = None
        for r in xl[sh]['A7':'A1000']:
            if r[0].value is None:
                nRows = i
                break
            i=i+1
        return nRows
    nRows = getNrows()
    jic_raw = pd.read_excel(fName,
                            sheet_name="1970=1 Infl Index",
                            header=0,usecols="A:BD",skiprows=5,
                            engine="openpyxl" ,
                            nrows=getNrows("1970=1 Infl Index"))
    jic_combof = pd.read_excel(fName,
                               sheet_name="CombOutFac(COF)",
                               header=0,usecols="A:BD",skiprows=5,
                               engine="openpyxl",
                               nrows=getNrows("CombOutFac(COF)"))
    meta = (
        pd.read_excel(fName, sheet_name ="Titles", header=0, usecols = "B:G", skiprows = 9, engine="openpyxl")
            .rename(columns = {"Idx": "Service", "Perm": "ID", "ShortTitle":"Indice"})
            .assign(Version = os.path.basename(fName), Date = xl.properties.modified)
            )
    def forecast_years(s,forecast=True):
        vals =[]
        for i in range(len(s)):
            if not np.isnan(s.iloc[i]):
                vals.append(s.iloc[i])
            else:
                if forecast:
                    vals.append(vals[i-1] / vals[i-2] * vals[i-1])
                else:
                    vals.append(vals[i-1])
        return vals

    addYears = range(jic_raw.Year.iloc[-1], 2100) 
    jic_raw = jic_raw.append(pd.DataFrame({'Year':addYears}), ignore_index=True)
    jic_raw.iloc[:,1:] = jic_raw.iloc[:,1:].apply(forecast_years, axis=0)
    jic_raw = jic_raw.melt(id_vars='Year',value_name='Raw', var_name='ID')

    jic_combof = jic_combof.append(pd.DataFrame({'Year':addYears}),ignore_index=True)
    jic_combof.iloc[:,1:] =jic_combof.iloc[:,1:].apply(forecast_years, forecast=False, axis=0) 
    jic_combof = jic_combof.melt(id_vars='Year',value_name='ComboF', var_name='ID')

    jic = pd.merge(jic_raw, jic_combof, how ='left', on=["Year", "ID"])
    jic = jic.assign(Weighted = jic.Raw * jic.ComboF)

    jic = meta.merge(jic, how='right', on="ID").drop_duplicates()
    jic.index = range(jic.shape[0])

    return jic

def change_BY(BY=2020, jic=None):
    if jic is None: jic = make_jic()
    by = jic[['ID','Year','Raw']].query("Year ==2020").rename(columns={"Raw":"BY_RAW"})
    new_jic = (
        pd.merge(jic, by[['ID','BY_RAW']], how='left', on='ID')
            .assign(Raw = new_jic.Raw / new_jic.BY_RAW, Weighted = new_jic.Raw / new_jic.BY_RAW * new_jic.ComboF)
            .drop(columns=["BY_RAW"])
        )

    return new_jic


def _get_jic():
    location = os.path.dirname(os.path.realpath(__file__))
    jic_file = os.path.join(location, 'data', 'inflation.csv')

    jic = pd.read_csv(jic_file)
    jic = stack_fys(jic, "Year")
    jic = jic.pivot_table(values=["value"], index=["Version", "Service", "tags", "Indice", "Long Title", "Year"],columns=["Type"]).reset_index()
    cols = []
    for n in jic.columns:
        if n[1] =="": 
            cols.append(n[0]) 
        else: 
            cols.append(n[1])
    jic.columns = cols
    return jic

jic = _get_jic()

def inflation(Index, FromYR, ToYR, Cost, from_type="BY", to_type="BY", inflation_table=None):
    '''
    Master inflation function
    '''
    if inflation_table is None:
        inflation_table = jic

    # 'nToYR = ToYR
    # If(ToYR > max(jic$Year)){ToYR <- max(jic$Year)}
    if type(Index) == str: 
        Index = [Index]
   
    df = pd.DataFrame({"Index": Index}).assign(FromYR = FromYR, ToYR=ToYR, Cost=Cost)

#    df = pd.DataFrame({"Index": Index, "FromYR": FromYR,
#                       "ToYR": ToYR, "Cost": Cost})

    # convert FY's to string to help look up values
    df["FromYR"] = df["FromYR"].map(str)
    df["ToYR"] = df["ToYR"].map(str)

    df_from = df.merge(inflation_table, how='left', left_on=[
                       "Index", "FromYR"], right_on=["Indice", "Year"])

    df_to = df.merge(inflation_table, how='left', left_on=[
                     "Index", "ToYR"], right_on=["Indice", "Year"])

    if from_type == "BY":
        div_by = df_from["Raw"].values
    else:
        div_by = df_from["Weighted"].values

    if to_type == "BY":
        mult_by = df_to["Raw"].values
    else:
        mult_by = df_to["Weighted"].values

    result = (df["Cost"].values / div_by) * mult_by
    #result = result[[1]]

    return result


# ' Title
# '
# ' @param Index
# ' @param FromYR
# ' @param ToYR
# ' @param Cost
# ' @param inflation_table
# '
# ' @return
# ' @export
# '
# ' @examples
def BYtoBY(Index, FromYR, ToYR, Cost, inflation_table=None):
    return inflation(Index, FromYR, ToYR, Cost, "BY", "BY", inflation_table)


# ' Title
# '
# ' @param Index
# ' @param FromYR
# ' @param ToYR
# ' @param Cost
# ' @param inflation_table
# '
# ' @return
# ' @export
# '
# ' @examples
def BYtoTY(Index, FromYR, ToYR, Cost, inflation_table=None):
    return inflation(Index, FromYR, ToYR, Cost, "BY", "TY", inflation_table)


# ' Title
# '
# ' @param Index
# ' @param FromYR
# ' @param ToYR
# ' @param Cost
# ' @param inflation_table
# '
# ' @return
# ' @export
# '
# ' @examples
def TYtoBY(Index, FromYR, ToYR, Cost, inflation_table=None):
    return inflation(Index, FromYR, ToYR, Cost, "TY", "BY", inflation_table)


# ' Title
# '
# ' @param Index
# ' @param FromYR
# ' @param ToYR
# ' @param Cost
# ' @param inflation_table
# '
# ' @return
# ' @export
# '
# ' @examples
def TYtoTY(Index, FromYR, ToYR, Cost, inflation_table=None):
    return inflation(Index, FromYR, ToYR, Cost, "TY", "TY", inflation_table)
