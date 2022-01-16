import scipy.stats
import scipy.optimize as opt
import numpy as np
import pandas as pd

from numbers import Real, Rational
__all__=['GlobalClock', 'SimEngine', 'RV', 'RVLognormal']

__variables__ = []
___engines__=[]

class GlobalClock:
    simulate = False
    random=False
    trial = 1
    trials = 1000

    
def find_log(x, mean=1, cv=.25, median=None, dist=scipy.stats.lognorm):
    obj = dist(*x)
    if median is None:
        m = abs(obj.mean()-mean)
    else:
        m = abs(obj.median()-median)
    
    obj_cv = obj.std() / obj.mean()
    cv = abs(obj_cv - cv)
    
    return m + cv



class RV(float,Real):
    '''
    RV is a Random Number generator that extends the float class. It is a special case
    where all math functions work. However, when setting the paramater simulate = True
    will provide random number generator following the distribution specificed.
    '''

    def __new__(self, mean=1, cv=.25, median = None, default_value = None, size=1, dist='lognorm', seed=None,simulate=False,random=False, engine=GlobalClock):
        if median is not None:
            _factor = median
            args = (None,  cv, 1)
        elif mean is not None:
            _factor = mean
            args = (1, cv, None)

        return float.__new__(self, _factor)

    def __init__(self, mean=1, cv=.25, median = None, default_value = None, size=1, dist='lognorm', seed=None, simulate=False, random=False, engine=GlobalClock):
        #if sum(x is not None for x in [mean, std, median]) != 2: retu
        self.engine = engine
        self.simulate=simulate
        self.random=random
        self.trial = 1
        self.seed = seed
        if median is not None:
            self._factor = median
            args = (None,  cv, 1)
        elif mean is not None:
            self._factor = mean
            args = (1, cv, None)


        x = opt.minimize(find_log, [1,0,1], args=args, method='Nelder-Mead',tol=1e-15, options = {'ftol':1e-15}) #method='SLSQP'
        if x.success:
            self.obj = scipy.stats.lognorm(*x.x)
            self.obj.random_state= np.random.default_rng(seed)
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
        simulate = self.engine.simulate
        random = self.engine.random
        trial = self.engine.trial
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
        trials = self.engine.trials
        self.rvs = self.obj.rvs(size=trials)


    def __str__(self):
        return str(self.get_value())
    def __repr__(self):
        return str(self.get_value())

    def __add__(self, rhs):
        if isinstance(rhs, RV): rhs = rhs.get_value()
        return self.get_value() + rhs
    __radd__ = __add__

    def __sub__(self, rhs):
        if isinstance(rhs, RV): rhs = rhs.get_value()
        return self.get_value() - rhs
    def __rsub__(self, lhs):
        if isinstance(lhs, RV): lhs = lhs.get_value()
        return lhs - self.get_value()


    def __mul__(self,rhs):
        if isinstance(rhs, RV): rhs = rhs.get_value()
        return self.get_value() * rhs
    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, RV): other = other.get_value()
        return other / self.get_value()

    def __rtruediv__(self,other):
        if isinstance(other, RV): other = other.get_value()
        return self.get_value() / other

    def __floordiv__(self, other):
        pass
    def __mod__(self, other):
        if isinstance(other, RV): other = other.get_value()
        return self.get_value() % other
    def __rfloordiv__(self, other):
        pass
    def __rmod__(self, other):
        if isinstance(other, RV): other = other.get_value()
        return other % self.get_value()
    def __divmod__(self, other):
        if isinstance(other, RV): other = other.get_value()
        pass
    def __pow__(self, other):
        if isinstance(other, RV): other = other.get_value()
        return self.get_value() * other

    def __rpow__(self, other):
        if isinstance(other, RV): other = other.get_value()
        return other ** self.get_value()

    def __abs__(self):
        return abs(self.get_value())

    def __eq__(self, other):
        if isinstance(other, RV): other = other.get_value()
        return self.get_value() == other
    def __gt__(self, other):
        if isinstance(other, RV): other = other.get_value()
        return self.get_value() > other
    def __ge__(self, other):
        if isinstance(other, RV): other = other.get_value()
        return get_value() >= other
    def __lt__(self, other):
        if isinstance(other, RV): other = other.get_value()
        return self.get_value() < other
    def __le__(self, other):
        if isinstance(other, RV): other = other.get_value()
        return self.get_value() <= other


    def __neg__(self):
        return - self.get_value()
    def __pos__(self):
        return self.get_value()

    def __int__(self):
        return int(self.get_value())
    def __float__(self):
        return float(self.get_value())

    def __round__(self, ndigits=0):
        return round(self.get_value(), ndigits)
    def __trunc__(self):
        return trunc(self.get_value())
    def __floor__(self):
        return floor(self.get_value())
    def __ceil__(self):
        return ceil(self.get_value())
    def __complex__(self):
        return complex(self.get_value())


class RVLognormal(RV):
    def __init__(self, mean=1, cv=.25,median=None, dist='lognorm'):
        super().__init__(mean=mean, cv = cv, median=median)




class SimEngine:
    def __init__(self,  trials=100,simulate=False,random=False, func=None, outputs=None, seed=None):
        self.simulate=simulate
        self.random=random
        self.trials = trials
        self.trial = 1
        self._variables = []
        self.correl_matrix = None
        if isinstance(seed,np.random.Generator):
            self.seed=seed
        else:
             self.seed=np.random.default_rng(seed)
    
    
    def run_simulation(self, func=lambda x:1, outputs=None):
        pass
        for i in range(self.trials):
            self.trial=i+1
            func()

    def RV(self, mean=1, cv=.25, median = None, default_value = None, size=1, dist='lognorm', seed=None):
        if seed is None:
            seed = abs(int(self.seed.normal(1000000,100000,size=1)))
        new_rv = RV( mean, cv, median , default_value, size, dist, seed,engine=self)
        self._variables.append(new_rv)
        return new_rv
   
    @staticmethod
    def correlate(rvs:list, correl_matrix=None, base_correl = .3, method='choleskly'):
        from scipy.linalg import eigh, cholesky

        # Correlation matrix
        if correl_matrix is None:
            n_rvs =len(rvs)
            correl_matrix = np.full((n_rvs,n_rvs), base_correl)
            np.fill_diagonal(correl_matrix, 1)

        if method == 'cholesky':
            # Compute the Cholesky decomposition.
            c = cholesky(correl_matrix, lower=True)
        else:
            # Compute the eigenvalues and eigenvectors.
            evals, evecs = eigh(correl_matrix)
            # Construct c, so c*c^T = r.
            c = np.dot(evecs, np.diag(np.sqrt(evals)))


        # Generate series of normally distributed (Gaussian) numbers
        rnd = np.vstack(rvs)
        y = np.dot(c, rnd)

        return y     

    @staticmethod
    def correlate_rvs(objs, correl_matrix=None, base_correl = .3, method='choleskly'):
        for rv in objs:
            if len(rv.rvs) == 0: rv.build_rvs() 
        rvs = [rv.rvs for rv in objs]
        new_rvs = SimEngine.correlate(rvs=rvs ,correl_matrix=correl_matrix, base_correl=base_correl,method=method)
        for i, obj in enumerate(objs):
            obj.rvs = new_rvs[i]

    
        
