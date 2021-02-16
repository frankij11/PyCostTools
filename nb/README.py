# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Py Cost Tools
# A suite of tools to help cost estimators
# 
# ## Installation
# * requires a copy of git 
# * otherwise need to clone / copy to desktop then run setup.py
# 
# ```
# pip install git+https://github.com/frankij11/PyCostTools.git#egg=pycost
# ```
# %% [markdown]
# ## Inflation
# * use the inflation functions as a simple calculator

# %%
import pycost as ct

print("BY20 to BY25", ct.BYtoBY(Index = "APN", FromYR = 2020, ToYR = 2025, Cost = 1))
print("TY25 to BY20", ct.TYtoBY(Index = "APN", FromYR = 2025, ToYR = 2020, Cost = 1))
print("Multiple Values", ct.BYtoBY(Index =['APN',"APN",'APN'], FromYR = [2020,2021,2022], ToYR = 2023, Cost=1))

# %% [markdown]
# * or use the inflation function as part of some analysis
# * works well with DataFrames

# %%
import pandas as pd
df = pd.DataFrame({'Index': ['APN']*10, 'Fiscal_Year':range(2020,2030), 'BY20': 1 })
df


# %%
df.assign(TY_DOL = lambda x: ct.BYtoTY(Index = x.Index,FromYR = 2020, ToYR=x.Fiscal_Year, Cost = x.BY20))

# %% [markdown]
# ## Analysis
# * helper functions to build models
# * comes preloaded with several sklearn models
# 
# As an example let's use data from the Joint Inflation Calculator to predict inflation
# %% [markdown]
# ### Toy Example: basics model flow
# 1. Define Model with data, formula, model (other specifications available)
# 1. Fit Model
# 1. View Summary
# 1. View Report
# 1. Make Predications
# 1. Save Model for later

# %%
# Import Libraries
from pycost.analysis import Model, Models, AutoRegressionLinear
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV


# %%
df = pd.DataFrame({'y': [1.5,2.2,3.2,4.9,5.0], 'x1': [2,4,6,8,10], 'x2': ["a", "b","b","a","a"]})
myModel = Model(df, "y~x1", model= LinearRegression(),test_split=0,
        meta_data={
            'title': "Example Analysis",
            'desc': "Do some anlaysis",
            'analyst': 'Kevin Joy',
            'FreeFileds': "Make whatever you like to doucment analysis"}
            )
myModel.fit().summary()


# %%
# show interactive report with fit statistics
myModel_report = myModel.report(False)
myModel_report


# %%
df = ct.jic.assign(Year=lambda x: pd.to_numeric(x.Year, 'coerce') ) # The year variable in jic is read as a string so must be converted
manyModels = Models(df = df,formulas="Raw~Year",by=['Service'], test_split=-1)
manyModels.fit()


# %%
manyModels.summary()


# %%
myModel.summary()


# %%
autoModel = AutoRegressionLinear(n_iter=10)
autoModel.fit(X=df.drop('Raw',axis=1), y=df.Raw)
autoModel.summary()


# %%



