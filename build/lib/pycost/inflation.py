import pandas as pd
import os
from .utils import *



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
        div_by = df_from["Raw"]
    else:
        div_by = df_from["Weighted"]

    if to_type == "BY":
        mult_by = df_to["Raw"]
    else:
        mult_by = df_to["Weighted"]

    result = (df["Cost"] / div_by) * mult_by
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
