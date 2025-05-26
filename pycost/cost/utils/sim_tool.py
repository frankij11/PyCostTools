import scipy.stats
import scipy.optimize as opt
import numpy as np
import pandas as pd
import itertools

# Try to import MCERP for UncertainFunction, used in plotcorr
try:
    import mcerp
    from mcerp import UncertainFunction
except ImportError:
    # Define a dummy class if MCERP is not available
    class UncertainFunction:
        pass

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

        self.obj = self._create_rv(mean=mean,median=median,cv=cv)
        self.obj.random_state= np.random.default_rng(seed)
        #x = opt.minimize(find_log, [1,0,1], args=args, method='Nelder-Mead',tol=1e-15, options = {'ftol':1e-15}) #method='SLSQP'
        #if x.success:
        #    self.obj = scipy.stats.lognorm(*x.x)
        #    self.obj.random_state= np.random.default_rng(seed)
        #else:
        #    print("could not find solution:/n", x)

        self._size= size
        self._dist = dist

        if default_value is None: 
            self.default_value = self.obj.mean() * self._factor
        else:
            self.default_value = default_value

        self.value = self.default_value
        self.rvs = []

    def _create_rv(self,mean=None, median=None,cv=None):
        
        if median and cv:
            mu = np.log(median)
            s2 = np.exp(np.log(1 + cv**2)/2)
        elif mean and cv:
            std = mean * cv
            variance = std**2
            s2 = np.log(variance/mean**2 + 1)
            mu = np.log(mean) - s2/2
        else:
            pass
        
        return scipy.stats.lognorm(scale=np.exp(mu), s = s2**.5)
    def plt(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        x = np.linspace(self.obj.ppf(0.01), self.obj.ppf(0.99))
        ax[0].plot(x, self.obj.pdf(x),'r-', lw=5, alpha=0.6, label='pdf') # x*self._factor
        ax[1].plot(x, self.obj.cdf(x),'k-', lw=5, alpha=0.6, label='cdf') # x*self._factor

    def get_value(self):
        simulate = self.engine.simulate
        random = self.engine.random
        trial = self.engine.trial
        if random:
            return self.obj.rvs(size=1)[0] #* self._factor
        elif simulate:
            if len(self.rvs) <self.engine.trials: self.build_rvs()
            return self.rvs[trial-1] #* self._factor
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
        return -self.get_value()
    def __pos__(self):
        return +self.get_value()

    def __int__(self):
        return int(self.get_value())
    def __float__(self):
        return float(self.get_value())

    def __round__(self, ndigits=0):
        return round(self.get_value(), ndigits)
    def __trunc__(self):
        return int(self.get_value())
    def __floor__(self):
        import math
        return math.floor(self.get_value())
    def __ceil__(self):
        import math
        return math.ceil(self.get_value())
    def __complex__(self):
        return complex(self.get_value())


class RVLognormal(RV):
    def __init__(self, mean=1, cv=.25,median=None, dist='lognorm'):
        super().__init__(mean=mean, cv=cv, median = median, dist=dist)


class SimEngine:
    def __init__(self,  trials=100,simulate=False,random=False, func=None, outputs=None, seed=None):
        self.simulate = simulate
        self.random = random
        self.rvs = []
        self.trial = 1
        self.trials = trials

        # Add this engine to the list of engines
        ___engines__.append(self)

        if callable(func):
            self.run_simulation(func, outputs)

    def run_simulation(self, func=lambda:1, outputs=None):
        #pass
        self.trials = 1000
        self.simulate = True
        res = []
        for i in range(1, self.trials+1):
            self.trial = i
            res.append(func())

        self.simulate = False
        self.res = res
        return res

    def RV(self, mean=1, cv=.25, median = None, default_value = None,size=1, dist='lognorm', seed=None):
        x = RV(mean=mean, cv=cv, median = median, default_value = default_value, size=size, dist=dist, seed=seed, engine=self)
        self.rvs.append(x)
        return x

    @staticmethod
    def correlate(rvs:list, correl_matrix=None, base_correl = .3, method='choleskly'):
        # Convert rvs to uniform random variables
        unifs = []
        for rv in rvs:
            x = rv.obj.cdf(rv.rvs)
            unifs.append(x)
            
        # Create base correlation matrix if not provided
        if correl_matrix is None:
            n = len(rvs)
            correl_matrix = np.full((n,n), base_correl)
            np.fill_diagonal(correl_matrix, 1)
            
        # Generate correlated standard normal samples
        samples = np.array(unifs).T
        samples = SimEngine.induce_correlations(samples, correl_matrix)
        samples = samples.T
            
        # Convert back from uniform to the original distributions
        for i, rv in enumerate(rvs):
            rv.rvs = rv.obj.ppf(samples[i])

    @staticmethod
    def correlate_rvs(objs, correl_matrix=None, base_correl = .3, method='choleskly'):
        """Correlate the random variables in the simulation
        correl_matrix - correlation matrix where (i, j) corresponds to correlation between objs[i].rvs and objs[j].rvs
        For now it's enforcing the correlation between the MCERP objects.
        Note that this function changes the mcerp objects.
        """
        if not all([hasattr(obj, "_mcpts") for obj in objs]):
            raise Exception("correlate_rvs only works on UncertainFunction objects")
            
        if correl_matrix is None:
            n = len(objs)
            correl_matrix = np.full((n,n), base_correl)
            np.fill_diagonal(correl_matrix, 1)
            
        all_samples = [obj._mcpts for obj in objs]
        samples = np.array(all_samples).T
        samples = SimEngine.induce_correlations(samples, correl_matrix)
        samples = samples.T
        
        # Now update the objects with the correlated samples
        for i, obj in enumerate(objs):
            obj._mcpts = samples[i]

    @staticmethod
    def induce_correlations(data, corrmat):
        """
        Induce a specified correlation matrix on a set of variables.
        
        Args:
            data: Input data array, shape (n_samples, n_variables)
            corrmat: Target correlation matrix, shape (n_variables, n_variables)
                
        Returns:
            Transformed data with the specified correlation matrix
            
        This method uses the Cholesky decomposition to induce correlations
        between variables.
        """
        # First convert the data to standard normal samples
        from scipy.stats import norm
        
        # Convert data to ranks
        ranks = np.zeros_like(data)
        for j in range(data.shape[1]):
            # Extract column and compute ranks
            col = data[:, j]
            ranks[:, j] = np.argsort(np.argsort(col)) / float(len(col))
        
        # Get standard normal samples based on ranks
        nsamps = norm.ppf(ranks)
        
        # Fill NaNs with 0 (happens when rank is exactly 0 or 1)
        nsamps[np.isnan(nsamps)] = 0
        
        # Compute correlation matrix
        obscorrmat = np.corrcoef(nsamps.T)
        
        # Use the Cholesky decomposition to create correlated samples
        try:
            obscholmat = SimEngine.chol(obscorrmat)
            cholmat = SimEngine.chol(corrmat)
        except np.linalg.LinAlgError:
            # Try with a small regularization if the matrix is not positive definite
            obscorrmat += 1e-10 * np.eye(obscorrmat.shape[0])
            corrmat += 1e-10 * np.eye(corrmat.shape[0])
            obscholmat = SimEngine.chol(obscorrmat)
            cholmat = SimEngine.chol(corrmat)
        
        # Transform the samples
        fnvars = np.matmul(np.matmul(nsamps, np.linalg.inv(obscholmat).T), cholmat.T)
        
        # Convert back to uniform ranks
        ranks = norm.cdf(fnvars)
        
        # Return data with original marginal distributions but the specified correlation
        newdata = np.zeros_like(data)
        for j in range(data.shape[1]):
            col = data[:, j]
            newdata[:, j] = np.sort(col)[np.floor(ranks[:, j] * len(col)).astype(int)]
        
        return newdata

    @staticmethod
    def plotcorr(X, plotargs=None, full=True, labels=None):
        """
        Plot the correlation matrix of a dataset.
        
        Args:
            X: Input data array, shape (n_samples, n_variables)
            plotargs: Dictionary of keyword arguments for plt.imshow()
            full: Whether to show the full correlation matrix or just the lower triangle
            labels: Labels for the variables
            
        Returns:
            The figure and axes objects
            
        This method creates a heatmap visualization of the correlation matrix
        between variables in the dataset.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Matplotlib is required for plotting")
            return None
        
        # Convert to numpy array if not already
        if isinstance(X, list):
            X = np.array(X).T
        
        # For UncertainFunction objects, extract the samples
        if X.ndim == 1 and isinstance(X[0], UncertainFunction):
            X = np.array([x._mcpts for x in X]).T
        
        # Calculate the correlation matrix
        corr = np.corrcoef(X, rowvar=False)
        
        # Handle default plotting arguments
        if plotargs is None:
            plotargs = {
                'cmap': plt.get_cmap('coolwarm'),
                'vmin': -1,
                'vmax': 1
            }
        
        # Set up the figure and axes
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Show only the lower triangle if requested
        if not full:
            corr = np.tril(corr)
        
        # Create the heatmap
        im = ax.imshow(corr, **plotargs)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Correlation')
        
        # Set labels
        if labels is None:
            if hasattr(X, 'columns'):  # For pandas DataFrame
                labels = X.columns
            else:
                labels = [f"Var {i+1}" for i in range(corr.shape[0])]
        
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        
        # Add correlation values as text
        for i in range(len(labels)):
            for j in range(len(labels)):
                if full or j <= i:
                    ax.text(j, i, f"{corr[i, j]:.2f}",
                           ha="center", va="center", 
                           color="white" if abs(corr[i, j]) > 0.5 else "black")
        
        ax.set_title("Correlation Matrix")
        fig.tight_layout()
        
        return fig, ax

    @staticmethod
    def chol(A):
        """
        Compute the Cholesky decomposition of a matrix.
        
        Args:
            A: Input matrix, must be positive definite
            
        Returns:
            The upper triangular Cholesky factor
            
        This method is a thin wrapper around numpy's Cholesky decomposition
        that handles some error cases.
        """
        try:
            return np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            # Try a regularized version if the matrix is not positive definite
            A_reg = A + 1e-8 * np.eye(A.shape[0])
            return np.linalg.cholesky(A_reg) 