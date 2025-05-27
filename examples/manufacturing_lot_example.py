import pandas as pd
import numpy as np
from pycost.learn import lc_prep, lc

import warnings
# To manage warnings, use the 'warnings' module. For example, to see warnings once:
# import warnings
# warnings.simplefilter('once') # Other options: 'default', 'error', 'always', etc.

# Create manufacturing lot data for 8 lots with columns:
# - Lot Type (EMD, LRIP, FRP)
# - FY (Fiscal Year)
# - LotQuantity

# Create the lot data
def create_lot_data(number_of_lots=8, total_quantity=100, lrip_percentage=0.10, T1=100, LC=0.95, RC=0.95, noise=0.05):
    '''Create Fake Lot Data for Testing that is typical of what we see in the real world'''
   
    emd_quantity = 2
    lrip_lots = 3
    frp_lots = number_of_lots - lrip_lots-1

    lrip_quantity = np.round((total_quantity-emd_quantity) * lrip_percentage, 0)
    print(lrip_quantity)
    lrip =[]
    cum_lrip = 0
    lrip= np.linspace(2, lrip_quantity, lrip_lots)
    #for i, lot in enumerate(lrip):
    #    lot_quantity = np.round(lot, 0)
    #    print("lot:", i, "quantity:", lot_quantity)
    #    if i == lrip_lots-1:
    #        lot_quantity = lrip_quantity-cum_lrip
    #    lrip[i]=lot_quantity
    #    cum_lrip += lot_quantity

    frp_quantity = np.round(total_quantity- lrip_quantity-emd_quantity, 0)

    print(emd_quantity, lrip_quantity, frp_quantity, sum([emd_quantity, lrip_quantity, frp_quantity]))

    # lrip should increase each year for the first 3 years
    #lrip = [np.round(lrip_quantity/lrip_lots-lrip_lots, 1)]*lrip_lots
    
    # frp should incremenet by 1 each year
    frp_beg = np.round(frp_quantity/frp_lots-frp_lots, 0)
    frp_end = np.round(frp_quantity/frp_lots, 0)
    frp = np.arange(int(frp_beg), int(frp_end), 1)
    
    
    quantity = [emd_quantity] + list(lrip) + list(frp)
    fy = np.arange(2020, 2020+number_of_lots, 1)
    lot_type = ['EMD'] + ['LRIP']*lrip_lots + ['FRP']*frp_lots

    print(len(quantity), len(fy), len(lot_type))
    print(quantity, fy, lot_type)
    df = pd.DataFrame(dict(
        program=['program x']*number_of_lots,
        quantity = quantity,
        fy = fy,
        lot_type = lot_type
    ))

    df = lc_prep(df, cols=['program', 'fy'], val='quantity', lc_slope=0.95)
    print(df)
    true_T1, true_b, true_c = T1, np.log(LC)/np.log(2), np.log(RC)/np.log(2)
    df["cost"] = true_T1 * df.midpoint**true_b * df.quantity**true_c * np.random.normal(1, noise, number_of_lots)

    return df

# Print the original data
df = pd.concat([create_lot_data(), create_lot_data().assign(program='program y')])

# Use lc_prep function to calculate first unit, last unit, and midpoint
# Assuming learning curve slope of 0.85 (typical aerospace value)
#lc_df = lc_prep(df, cols=['Lot', 'Type', 'FY'], val='quantity', lc_slope=0.95)

print(df)
# plot the data
import seaborn as sns
import matplotlib.pyplot as plt
p=sns.scatterplot(x='midpoint', y='cost', data=df)
p.set_title('Cost vs Midpoint')
p.set_xlabel('Midpoint')
p.set_ylabel('Cost')
#plt.show()

# Print the expanded data with first unit, last unit, and midpoint calculations
print("Manufacturing Lot Data with Learning Curve Calculations:")
print(df[['program', 'fy', 'lot_type', 'quantity', 'midpoint']])

# Export the data to CSV for further use
#lc_df.to_csv('manufacturing_lots.csv', index=False)
#print("\nData exported to 'manufacturing_lots.csv'")

# Show example code for using this with your ConstrainedRegression model
# compare model types
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from pycost.analysis.constrained.constrained_model import ConstrainedRegression, ConstrainedRegressionCV
from pycost.analysis.model import Model, Models
alphas = np.arange(0.001,1.001,0.1)
models = {'linear': LinearRegression(), 'lasso': LassoCV(alphas=alphas), 'ridge': RidgeCV(alphas=alphas), 'elasticnet': ElasticNetCV(alphas=alphas), 
          'constrained': ConstrainedRegression(coef_bounds={'midpoint': (-.15,-.000001), 'quantity': (-.15,-.000001)}), 
          'constrained_ridge': ConstrainedRegressionCV(coef_bounds={'midpoint': (-.15,-.000001), 'quantity': (-.15,-.000001)},alphas=alphas,l1_ratios=[0]),
          'constrained_lasso': ConstrainedRegressionCV(coef_bounds={'midpoint': (-.15,-.000001), 'quantity': (-.15,-.000001)},alphas=alphas,l1_ratios=[1]),
          'constrained_enet': ConstrainedRegressionCV(coef_bounds={'midpoint': (-.15,-.000001), 'quantity': (-.15,-.000001)},alphas=alphas,l1_ratios=np.arange(0,1,0.1))}
X = df[['midpoint', 'quantity']]#.rename(columns={'quantity': 'QTY'})
y = df['cost']

X = np.log(X)
y = np.log(y)

for model in models:
    
    models[model].fit(X, y)
    print(model, 2**models[model].coef_, np.exp(models[model].intercept_))
    if hasattr(models[model], 'summary'):
        print(models[model].summary())

    # add predictions dataframe
    df['pred_'+model] = np.exp(models[model].predict(X))


lc_models = Models(df, formulas=['np.log(cost) ~ np.log(midpoint) + np.log(quantity)'], models=[models[model] for model in models], by='program')

lc_models.fit()

print(lc_models.summary())

print(lc_models.db.head(1).T)


print(df)
# plot all the predicted values vs the actual values on the same plot
plt.figure(figsize=(12, 8))
# Plot actual cost
plt.scatter(df['midpoint'], df['cost'], s=100, c='black', label='Actual Cost', zorder=10)

# Plot predictions with different colors
models = ['linear', 'lasso', 'ridge', 'elasticnet', 'constrained', 'constrained_ridge', 'constrained_lasso', 'constrained_enet']
colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown']

for i, model in enumerate(models):
    plt.scatter(df['midpoint'], df[f'pred_{model}'], alpha=0.7, label=f'{model.replace("_", " ").title()}', c=colors[i])

plt.xlabel('Midpoint')
plt.ylabel('Cost')
plt.title('Cost vs Midpoint: Actual vs Model Predictions')
plt.legend(loc='best')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.show()
