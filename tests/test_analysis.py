# %%
# import sys, os
# p = os.path.dirname(__file__)
# os.chdir(p)
# os.chdir("../pycost")
# #p = os.path.dirname(__file__)
# #print(p)
# #sys.path.insert(0, p)

# %%
import sklego  # .preprocessing.PatsyTransformer as PT
from pycost.analysis import *
import pycost as ct
import pandas as pd
df = ct.jic
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
f = "Raw ~ Version+Service+tags+Indice +Year"

# %%
m = Model(df, f).fit()
myModels = Models(df, "Raw ~ Year + tags + Indice",
                  by=["Version", "Service"], handle_na=False, tags={"JIC": 2019, "Analyst": "Kevin Joy"}).fit()


new_df = df.loc[0:2].assign(Year=range(2090, 2093))

#app,server = mods[0].report()


# %%
app = myModels.report(show=False)
app
# %%

#p = PT()
# %%
# %%
