import scipy.stats
import scipy.optimize as opt
import numpy as np
import pandas as pd
__all__=['simulate','random', 'trial', 'trials', 'RV', 'RVLognormal']

def find_log(x, mean=1, cv=.25, median=None, dist=scipy.stats.lognorm):
    obj = dist(*x)
    if median is None:
        m = abs(obj.mean()-mean)
    else:
        m = abs(obj.median()-median)
    
    obj_cv = obj.std() / obj.mean()
    cv = abs(obj_cv - cv)
    
    return m + cv

simulate = False
random=False
trial = 1
trials = 1000

class RV:

    
    def __init__(self, mean=1, cv=.25, median = None, default_value = None, size=1, dist='lognorm', seed=123):
        #if sum(x is not None for x in [mean, std, median]) != 2: retu
            
        if median is not None:
            self._factor = median
            args = (None,  cv, 1)
        elif mean is not None:
            self._factor = mean
            args = (1, cv, None)
        
        
        x = opt.minimize(find_log, [1,0,1], args=args, method='SLSQP', tol=None)
        if x.success:
            self.obj = scipy.stats.lognorm(*x.x)
        else:
            print("could not find solution:/n", x)

        self._size= size
        self._dist = dist
            
        if default_value is None: 
            self.default_value = self.obj.mean() * self._factor
        else:
            self.default_value = default_value
        
        self.value = self.default_value
        self.rvs = []
    
    def plt(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        x = np.linspace(self.obj.ppf(0.01), self.obj.ppf(0.99))
        ax[0].plot(x*self._factor, self.obj.pdf(x),'r-', lw=5, alpha=0.6, label='pdf')
        ax[1].plot(x*self._factor, self.obj.cdf(x),'k-', lw=5, alpha=0.6, label='cdf')
    
    def get_value(self):
        global simulate
        global random
        global trial
        if random:
            return self.obj.rvs(size=1)[0] * self._factor
        elif simulate:
            if len(self.rvs) ==0: self.build_rvs()
            return self.rvs[trial-1] * self._factor
        else:
            return self.value
    
    def reset_rvs(self):
        self.rvs = []
        #set seed

    def build_rvs(self):
        # set seed
        global trials
        self.rvs = self.obj.rvs(size=trials)
    
    def __add__(self, rhs):
        return self.get_value() + rhs
    __radd__ = __add__
    
    
class RVLognormal(RV):
    def __init__(self, mean=1, cv=.25,median=None, dist='lognorm'):
        super().__init__(mean=mean, cv = cv, median=median)
