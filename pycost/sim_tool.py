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


    def run_simulation(self, func=lambda:1, outputs=None):
        #pass
        self.simulate=True
        tmp = pd.DataFrame()
        for i in range(self.trials):
            self.trial=i+1
            #func()
            tmp = tmp.append(func().assign(trial=i+1),ignore_index=True)
        self.simulate=False
        return tmp

    def RV(self, mean=1, cv=.25, median = None, default_value = None,size=1, dist='lognorm', seed=None):
        if seed is None:
            seed = abs(int(self.seed.normal(1000000,100000,size=1)))
        
        new_rv = RV( mean, cv, median , default_value, size, dist, seed,engine=self)
        
        self._variables.append(new_rv)
        return new_rv

    @staticmethod
    def correlate(rvs:list, correl_matrix=None, base_correl = .3, method='choleskly'):
        from scipy.linalg import eigh, cholesky
        data = np.vstack(rvs).T

        # Correlation matrix
        if correl_matrix is None:
            n_rvs =len(rvs)
            correl_matrix = np.full((n_rvs,n_rvs), base_correl)
            np.fill_diagonal(correl_matrix, 1)

        if method == 'cholesky':
            # Compute the Cholesky decomposition.
            #c = cholesky(correl_matrix, lower=True)
            y = SimEngine.induce_correlations(data, correl_matrix)
        else:
            # Compute the eigenvalues and eigenvectors.
            evals, evecs = eigh(correl_matrix)
            # Construct c, so c*c^T = r.
            c = np.dot(evecs, np.diag(np.sqrt(evals)))
            rnd = np.vstack(rvs)
            y = np.dot(c, rnd)


        # Generate series of normally distributed (Gaussian) numbers


        return y     

    @staticmethod
    def correlate_rvs(objs, correl_matrix=None, base_correl = .3, method='choleskly'):
        for rv in objs:
            if len(rv.rvs) == 0: rv.build_rvs() 
        rvs = [rv.rvs for rv in objs]
        
         
        new_rvs = SimEngine.correlate(rvs=rvs ,correl_matrix=correl_matrix, base_correl=base_correl,method=method)
        for i, obj in enumerate(objs):
            obj.rvs = new_rvs[:,i]
            
    @staticmethod
    def induce_correlations(data, corrmat):
        """
        Induce a set of correlations on a column-wise dataset

        Parameters
        ----------
        data : 2d-array
            An m-by-n array where m is the number of samples and n is the
            number of independent variables, each column of the array corresponding
            to each variable
        corrmat : 2d-array
            An n-by-n array that defines the desired correlation coefficients
            (between -1 and 1). Note: the matrix must be symmetric and
            positive-definite in order to induce.

        Returns
        -------
        new_data : 2d-array
            An m-by-n array that has the desired correlations.

        """
        # Create an rank-matrix
        from scipy.stats import rankdata
        from scipy.stats.distributions import norm
        data_rank = np.vstack([rankdata(datai) for datai in data.T]).T

        # Generate van der Waerden scores
        data_rank_score = data_rank / (data_rank.shape[0] + 1.0)
        data_rank_score = norm(0, 1).ppf(data_rank_score)

        # Calculate the lower triangular matrix of the Cholesky decomposition
        # of the desired correlation matrix
        p = SimEngine.chol(corrmat)

        # Calculate the current correlations
        t = np.corrcoef(data_rank_score, rowvar=0)

        # Calculate the lower triangular matrix of the Cholesky decomposition
        # of the current correlation matrix
        q = SimEngine.chol(t)

        # Calculate the re-correlation matrix
        s = np.dot(p, np.linalg.inv(q))

        # Calculate the re-sampled matrix
        new_data = np.dot(data_rank_score, s.T)

        # Create the new rank matrix
        new_data_rank = np.vstack([rankdata(datai) for datai in new_data.T]).T

        # Sort the original data according to new_data_rank
        for i in range(data.shape[1]):
            vals, order = np.unique(
                np.hstack((data_rank[:, i], new_data_rank[:, i])), return_inverse=True
            )
            old_order = order[: new_data_rank.shape[0]]
            new_order = order[-new_data_rank.shape[0] :]
            tmp = data[np.argsort(old_order), i][new_order]
            data[:, i] = tmp[:]

        return data

    @staticmethod
    def plotcorr(X, plotargs=None, full=True, labels=None):
        """
        Plots a scatterplot matrix of subplots.  

        Usage:

            plotcorr(X)

            plotcorr(..., plotargs=...)  # e.g., 'r*', 'bo', etc.

            plotcorr(..., full=...)  # e.g., True or False

            plotcorr(..., labels=...)  # e.g., ['label1', 'label2', ...]
        Each column of "X" is plotted against other columns, resulting in
        a ncols by ncols grid of subplots with the diagonal subplots labeled 
        with "labels".  "X" is an array of arrays (i.e., a 2d matrix), a 1d array
        of MCERP.UncertainFunction/Variable objects, or a mixture of the two.
        Additional keyword arguments are passed on to matplotlib's "plot" command. 
        Returns the matplotlib figure object containing the subplot grid.
        """
        import matplotlib.pyplot as plt

        X = [Xi._mcpts if isinstance(Xi, UncertainFunction) else Xi for Xi in X]
        X = np.atleast_2d(X)
        numvars, numdata = X.shape
        fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.0, wspace=0.0)

        for ax in axes.flat:
            # Hide all ticks and labels
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            # Set up ticks only on one side for the "edge" subplots...
            if full:
                if ax.is_first_col():
                    ax.yaxis.set_ticks_position("left")
                if ax.is_last_col():
                    ax.yaxis.set_ticks_position("right")
                if ax.is_first_row():
                    ax.xaxis.set_ticks_position("top")
                if ax.is_last_row():
                    ax.xaxis.set_ticks_position("bottom")
            else:
                if ax.is_first_row():
                    ax.xaxis.set_ticks_position("top")
                if ax.is_last_col():
                    ax.yaxis.set_ticks_position("right")

        # Label the diagonal subplots...
        if not labels:
            labels = ["x" + str(i) for i in range(numvars)]

        for i, label in enumerate(labels):
            axes[i, i].annotate(
                label, (0.5, 0.5), xycoords="axes fraction", ha="center", va="center"
            )

        # Plot the data
        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            if full:
                idx = [(i, j), (j, i)]
            else:
                idx = [(i, j)]
            for x, y in idx:
                # FIX #1: this needed to be changed from ...(data[x], data[y],...)
                if plotargs is None:
                    if len(X[x]) > 100:
                        plotargs = ",b"  # pixel marker
                    else:
                        plotargs = ".b"  # point marker
                axes[x, y].plot(X[y], X[x], plotargs)
                ylim = min(X[y]), max(X[y])
                xlim = min(X[x]), max(X[x])
                axes[x, y].set_ylim(
                    xlim[0] - (xlim[1] - xlim[0]) * 0.1, xlim[1] + (xlim[1] - xlim[0]) * 0.1
                )
                axes[x, y].set_xlim(
                    ylim[0] - (ylim[1] - ylim[0]) * 0.1, ylim[1] + (ylim[1] - ylim[0]) * 0.1
                )

        # Turn on the proper x or y axes ticks.
        if full:
            for i, j in zip(list(range(numvars)), itertools.cycle((-1, 0))):
                axes[j, i].xaxis.set_visible(True)
                axes[i, j].yaxis.set_visible(True)
        else:
            for i in range(numvars - 1):
                axes[0, i + 1].xaxis.set_visible(True)
                axes[i, -1].yaxis.set_visible(True)
            for i in range(1, numvars):
                for j in range(0, i):
                    fig.delaxes(axes[i, j])

        # FIX #2: if numvars is odd, the bottom right corner plot doesn't have the
        # correct axes limits, so we pull them from other axes
        if numvars % 2:
            xlimits = axes[0, -1].get_xlim()
            ylimits = axes[-1, 0].get_ylim()
            axes[-1, -1].set_xlim(xlimits)
            axes[-1, -1].set_ylim(ylimits)

        return fig

    @staticmethod
    def chol(A):
        """
        Calculate the lower triangular matrix of the Cholesky decomposition of
        a symmetric, positive-definite matrix.
        """
        A = np.array(A)
        assert A.shape[0] == A.shape[1], "Input matrix must be square"

        L = [[0.0] * len(A) for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                L[i][j] = (
                    (A[i][i] - s) ** 0.5 if (i == j) else (1.0 / L[j][j] * (A[i][j] - s))
                )

        return np.array(L)
