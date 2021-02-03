# %%
import sys, os
p = os.path.dirname(__file__)
print(p)
sys.path.insert(0, p)

# %%
from analysis import Model
import pycost as ct
import pandas as pd
df = ct.jic
df['Year'] = pd.to_numeric( df['Year'], errors='coerce')
f = "Raw ~ Version+Service+tags+Indice +Year"
model_types = dict(
                    lm = LinearRegression(),
                    rf=RandomForestRegressor(), 
                    lasso=LassoCV(cv=5), 
                    ridge=RidgeCV(cv=5), 
                    enet = ElasticNetCV(cv=5))
mods = []
results = pd.DataFrame()
for mod in model_types:    
    m=Model(df=ct.jic, formula=f, model=model_types[mod])
    m.fit()
    #print(m.feature_names)
    #print(m.summary().T)
    mods.append(m)
    results = pd.concat([results, m.summary()], ignore_index=True)

print(results.T)
#print(mods[0])

new_df = df.loc[0:2].assign(Year=range(2090,2093))
for mod in mods:
    print(mod.model['model'],mod.predict(new_df))
    print("closest points")
    print([item for key, item in mod.find_knn(new_df, 5).items()])
#app,server = mods[0].report()


