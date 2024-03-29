from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from sklearn import metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LearnCurve:
    def __init__(self) -> None:
        pass
    
    def fit(self, X,y, **kwargs):
        self.model = self.lc_reg(self.func, X,y, **kwargs)
        #self.T1,self.LC, self.RC = (*self.model['params'])
        return self

    def predict(self,X):
        y=self.func(X,*self.model['params'])
        return y

    @staticmethod
    def func(x, T1, LC, RC, **kwargs):
        b = np.log(LC) / np.log(2)
        r = np.log(RC) / np.log(2)
        ans = T1*(x.midpoint **b) * (x.QTY**r)
        for key, value in kwargs.items():
            ans = ans * x[key] * value
            print("The value of {} is {}".format(key, value))
        return ans
        
    @staticmethod
    def lc_reg(func, xdata, ydata, bounds=([0, .7,.7], [np.inf, 1, 1]), params=["T1", "LC", "RC"]):
        

        # fit regression
        popt, pcov = curve_fit(func, xdata, ydata, bounds=bounds)

        # standard deviation
        perr = np.sqrt(np.diag(pcov))

        # bounds of vars
        params = pd.DataFrame({"Param":params,'LB':popt -3*perr , 'Value': popt, 'UB': popt +3*perr})



        reg_stats = pd.DataFrame({'RSQ': [metrics.r2_score(ydata, func(xdata, *popt))]})
        reg_stats = reg_stats.assign(MAE = metrics.mean_absolute_error(ydata, func(xdata, *popt)),
                                    RMSE = metrics.mean_squared_error(ydata, func(xdata, *popt))**.5,
                                    Max_Error = metrics.max_error(ydata, func(xdata, *popt)),
                                    N_Obs = ydata.shape[0],
                                    df = ydata.shape[0] - popt.shape[0]
                                    )

        return {'params':popt, 'reg_stats': reg_stats, 'param_bounds':params}


if __name__ =='__main__':
    # generate fake data
    xdata = pd.DataFrame({'midpoint': np.arange(1,11), 'QTY': np.random.uniform(1,30,size=10)})
    y = LearnCurve.func(xdata, 100, .95, .83)
    np.random.seed(1729)
    y_noise =  np.random.normal(1, .10, size=xdata.shape[0])
    ydata = y * y_noise
    
    LC = LearnCurve()
    LC.fit(xdata,ydata)
    LC.predict(xdata)
    print(LC.model)
