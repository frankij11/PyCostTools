from scipy.optimize import least_squares, curve_fit, minimize

from patsy import dmatrices, build_design_matrices
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import libraries for sklearn mixins
from sklearn import metrics
from sklearn.base import BaseEstimator, RegressorMixin

# Create a constrained regression model that allows for bounds on coefficients
# This is compatible with sklearn pipelines
# Uses scipy.optimize.minimize to find the best coefficients

from typing import Callable, Optional, Dict, Tuple, Any, List, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import types


class DotDict(dict):
    """Simple dot-access dictionary."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{name}'")
    def __setattr__(self, name, value):
        self[name] = value
    def __delattr__(self, name):
        del self[name]


class ConstrainedRegression(BaseEstimator, RegressorMixin):
    """
    General regression with:
      - Coefficient constraints (bounds, partial via dict)
      - Custom prediction functions (linear default or user-provided)
      - Fit-intercept handling
      - L1/L2 regularization (alpha, l1_ratio)
      - Named-coefficient access (.coef_dict_.T1, .coef_dict_.b, etc.)
      - Score (R²), summary(), and coef_variance()
    """

    def __init__(
        self,
        coef_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        default_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        fit_intercept: bool = True,
        l1_ratio: float = 0.0,
        alpha: float = 0.0,
        minimize_kwargs: Optional[Dict[str, Any]] = None,
        func: Optional[Callable[[pd.DataFrame, Any], np.ndarray]] = None,
        param_names: Optional[List[str]] = None,
        use_smart_init: bool = True,
    ):
        self.coef_bounds    = coef_bounds
        self.default_bounds = default_bounds
        self.fit_intercept  = fit_intercept
        self.l1_ratio       = l1_ratio
        self.alpha          = alpha
        self.minimize_kwargs = minimize_kwargs #or {}
        self.param_names    = param_names
        self.use_smart_init = use_smart_init
        # Default to a linear model with intercept handling
        self.func = func #if func is not None else self._default_func

        # Fix Clone error where clone expects parameters to not be changed by __init__
        
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"alpha={self.alpha}, l1_ratio={self.l1_ratio}, "
                f"intercept={self.fit_intercept})")
    
    def _default_func(self, X: pd.DataFrame, coefs: DotDict) -> np.ndarray:
        """Linear: intercept + X @ coefficients."""
        intercept = coefs.intercept if self.fit_intercept else 0.0
        coefs_values =[]
        for col in X.columns:
            if col in coefs:
                coefs_values.append(coefs[col])
            else:
                coefs_values.append(0.0)
        coef_values = np.array(coefs_values)
        return intercept + X.values @ coef_values

    def _get_feature_names(self, X: Union[pd.DataFrame, np.ndarray]) -> List[str]:
        if isinstance(X, pd.DataFrame):
            return list(X.columns)
        return [f"x{i}" for i in range(X.shape[1])]

    def _wrap_func(self, raw_func: Callable, X: Any, coef_vector: np.ndarray):
        # Ensure DataFrame for column-based access
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        # Build named coefficient map
        if self.fit_intercept:
            names = ["intercept"] + (self.param_names or self.feature_names_in_)
        else:
            names = self.param_names or self.feature_names_in_
        coefs = DotDict(dict(zip(names, coef_vector)))
        return raw_func(X, coefs)
        
    def _compute_smart_init(self, X, y):
        """Get smart initial values via OLS/WLS with bounds projection."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        
        if self.func == self._default_func or self.func is None:
            # For linear models, use direct OLS
            if self.fit_intercept:
                # Add intercept column
                X_with_intercept = np.hstack([np.ones((X_arr.shape[0], 1)), X_arr])
                try:
                    # Use lstsq for numerical stability
                    init_coefs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                except np.linalg.LinAlgError:
                    # Fallback to zeros
                    init_coefs = np.zeros(X_arr.shape[1] + 1)
            else:
                try:
                    init_coefs = np.linalg.lstsq(X_arr, y, rcond=None)[0]
                except np.linalg.LinAlgError:
                    init_coefs = np.zeros(X_arr.shape[1])
        else:
            # For non-linear models, start with reasonable guess
            n_params = len(self.coef_names_)
            if self.func == self.func and hasattr(self, 'param_names') and self.param_names:
                # If it's a learning curve model or similar, use specific defaults
                if 'T1' in self.param_names:
                    # Estimate T1 as maximum y value
                    t1_idx = self.param_names.index('T1')
                    init_coefs = np.ones(n_params)
                    init_coefs[t1_idx] = max(y) * 1.2  # slightly higher than max
                else:
                    init_coefs = np.ones(n_params)
            else:
                init_coefs = np.ones(n_params)
        
        # Project onto bounds
        bounds = []
        for i, name in enumerate(self.coef_names_):
            if name == "intercept" and self.fit_intercept:
                bounds.append((None, None))
            else:
                bounds.append(self.coef_bounds.get(name, self.default_bounds)
                              if self.coef_bounds else self.default_bounds)
        
        # Apply bounds to initial coefficients
        for i, (lb, ub) in enumerate(bounds):
            if lb is not None and init_coefs[i] < lb:
                init_coefs[i] = lb
            if ub is not None and init_coefs[i] > ub:
                init_coefs[i] = ub
                
        return init_coefs

    def _compute_gradient(self, X, y, params):
        """Compute analytical gradient for linear model with regularization."""
        if self.func == self._default_func or self.func is None:
            # Linear model gradient: -2X'(y - Xβ) + regularization
            X_arr = X.values if isinstance(X, pd.DataFrame) else X
            if self.fit_intercept:
                X_with_intercept = np.hstack([np.ones((X_arr.shape[0], 1)), X_arr])
                preds = X_with_intercept @ params
                grad = -2 * X_with_intercept.T @ (y - preds)
            else:
                preds = X_arr @ params
                grad = -2 * X_arr.T @ (y - preds)
            
            # Add regularization gradient
            if self.alpha > 0:
                reg_params = params[1:] if self.fit_intercept else params
                reg_grad = np.zeros_like(params)
                
                if self.fit_intercept:
                    # Skip intercept in regularization
                    if self.l1_ratio > 0:  # L1 component
                        l1_grad = self.alpha * self.l1_ratio * np.sign(reg_params)
                        reg_grad[1:] += l1_grad
                    
                    if self.l1_ratio < 1:  # L2 component
                        l2_grad = self.alpha * (1 - self.l1_ratio) * 2 * reg_params
                        reg_grad[1:] += l2_grad
                else:
                    if self.l1_ratio > 0:  # L1 component
                        l1_grad = self.alpha * self.l1_ratio * np.sign(reg_params)
                        reg_grad += l1_grad
                    
                    if self.l1_ratio < 1:  # L2 component
                        l2_grad = self.alpha * (1 - self.l1_ratio) * 2 * reg_params
                        reg_grad += l2_grad
                
                grad += reg_grad
            
            return grad
        else:
            # For custom functions, use numerical gradient
            return None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray):
        # Preserve DataFrame columns if present
        if not self.func:
            self.func = self._default_func
        if not self.minimize_kwargs:
            self.minimize_kwargs = {}
        X_df = isinstance(X, pd.DataFrame)
        orig_columns = list(X.columns) if X_df else None

        X, y = check_X_y(X, y, dtype=float)
        if X_df:
            X = pd.DataFrame(X, columns=orig_columns)

        self.feature_names_in_ = self._get_feature_names(X)
        self._X_fit, self._y_fit = X, y

        # Define coefficient names (incl. intercept)
        if self.fit_intercept:
            coef_names = ["intercept"] + (self.param_names or self.feature_names_in_)
        else:
            coef_names = self.param_names or self.feature_names_in_
        self.coef_names_ = coef_names
        n_params = len(coef_names)

        # Objective with optional L1/L2 penalties
        def loss_fn(params):
            preds = self._wrap_func(self.func, X, params)
            resid = y - preds
            loss = np.sum(resid ** 2)
            if self.alpha > 0:
                c = params[1:] if self.fit_intercept else params
                l1 = self.l1_ratio * np.sum(np.abs(c))
                l2 = (1 - self.l1_ratio) * np.sum(c**2)
                loss += self.alpha * (l1 + l2)
            return loss

        # Initial guess + bounds
        if self.use_smart_init:
            x0 = self._compute_smart_init(X, y)
        else:
            x0 = np.zeros(n_params)
            
        bounds = []
        for name in coef_names:
            if name == "intercept":
                bounds.append((None, None))
            else:
                bounds.append(self.coef_bounds.get(name, self.default_bounds)
                              if self.coef_bounds else self.default_bounds)

        # Try to use analytical gradient for linear models
        jac = None
        is_linear = (self.func == self._default_func or self.func is None)
        if is_linear:
            jac = lambda params: self._compute_gradient(X, y, params)
            
        # Setup default method if not specified
        if 'method' not in self.minimize_kwargs:
            if is_linear:
                self.minimize_kwargs['method'] = 'L-BFGS-B'  # Good for linear with gradient
            else:
                self.minimize_kwargs['method'] = 'SLSQP'  # Good for nonlinear

        # Run optimization
        result = minimize(loss_fn, x0, bounds=bounds, jac=jac, **self.minimize_kwargs)
        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)
        self.result_ = result

        # Save intercept & coefficients
        p = result.x
        if self.fit_intercept:
            self.intercept_, self.coef_ = p[0], p[1:]
        else:
            self.intercept_, self.coef_ = 0.0, p
        self.coef_dict_ = DotDict(dict(zip(coef_names, result.x)))
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_df = isinstance(X, pd.DataFrame)
        orig_columns = list(X.columns) if X_df else None

        X = check_array(X, dtype=float)
        if X_df:
            X = pd.DataFrame(X, columns=orig_columns)

        vector = np.array([self.coef_dict_[n] for n in self.coef_names_])
        return self._wrap_func(self.func, X, vector)

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) -> float:
        """R² score."""
        y_pred = self.predict(X)
        return 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

    def coef_variance(self) -> np.ndarray:
        """Variances from Hessian inverse (if available)."""
        hess_inv = getattr(self.result_, "hess_inv", None)
        if hess_inv is None:
            raise RuntimeError("Hessian inverse unavailable.")
        mat = hess_inv.todense() if hasattr(hess_inv, "todense") else hess_inv
        return np.diag(mat)

    def summary(self):
        """Print coefficients ±σ, objective, and R²."""
        print("Model Summary:")
        print("-" * 40)
        try:
            variances = self.coef_variance()
        except RuntimeError:
            variances = None
        for i, name in enumerate(self.coef_names_):
            v = (f" ±{np.sqrt(variances[i]):.4f}" if variances is not None else "")
            print(f"{name:15s}: {self.coef_dict_[name]:.4f}{v}")
        print("-" * 40)
        print(f"Objective Value: {self.result_.fun:.4f}")
        print(f"R² Score:       {self.score(self._X_fit, self._y_fit):.4f}")

class LearnCurve(ConstrainedRegression):
    def __init__(self, **kwargs):
        kwargs.setdefault('param_names', ["T1", "LC", "RC"])
        kwargs.setdefault('coef_bounds', {"LC": (.7, 1), "RC": (.7, 1)})
        kwargs.setdefault('fit_intercept', False)
        kwargs.setdefault('minimize_kwargs', {"method": "L-BFGS-B"})
        kwargs.setdefault('use_smart_init', True)
        super().__init__(**kwargs)
        self.func = self.lc_func
    
    def lc_func(self, X, coefs):
        """Learning curve function: T1 * (midpoint^b) * (QTY^r) where b=log(LC)/log(2), r=log(RC)/log(2)"""
        b = np.log(coefs.LC) / np.log(2)
        r = np.log(coefs.RC) / np.log(2)
        ans = coefs.T1 * (X.midpoint**b) * (X.QTY**r)
        return ans
        
    def _compute_smart_init(self, X, y):
        """Special initialization for learning curve models."""
        # For LC model: reasonable defaults + scaling based on data
        # T1 is approx the "first unit cost"
        # Typical learning curve has LC around 0.85-0.95
        # Typical rate curve has RC around 0.85-0.95
        
        # Estimate T1 by scaling max value
        if len(y) > 0:
            # Find points with lowest midpoint values
            if 'midpoint' in X.columns:
                min_idx = X.midpoint.idxmin()
                if isinstance(min_idx, (list, np.ndarray)) and len(min_idx) > 0:
                    min_idx = min_idx[0]
                
                # Approximate T1 based on the smallest midpoint value and its corresponding y
                # Apply the inverse of the learning curve function
                try:
                    min_mid = X.loc[min_idx, 'midpoint']
                    min_qty = X.loc[min_idx, 'QTY']
                    min_y = y[min_idx]
                    
                    # Assume standard learning/rate curves (0.85)
                    b_est = np.log(0.85) / np.log(2)
                    r_est = np.log(0.85) / np.log(2)
                    
                    # T1 = y / (midpoint^b * qty^r)
                    t1_est = min_y / ((min_mid**b_est) * (min_qty**r_est))
                    # Add safety margin
                    t1_est *= 1.1
                except:
                    # Fallback if error
                    t1_est = np.max(y) * 1.5
            else:
                t1_est = np.max(y) * 1.5
                
            coefs = [t1_est, 0.85, 0.85]  # [T1, LC, RC]
        else:
            coefs = [100.0, 0.85, 0.85]  # Default values
            
        return np.array(coefs)


class OLD_LearnCurve:
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

from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin

class ConstrainedRegressionCV(BaseEstimator, RegressorMixin):
    """
    Cross-validated wrapper for ConstrainedRegression over a grid of alpha (regularization strength)
    and l1_ratio (L1/L2 mix) hyperparameters.

    Parameters
    ----------
    coef_bounds : dict, optional
        Bounds for each coefficient passed to ConstrainedRegression.

    default_bounds : tuple, default=(None, None)
        Default bound for any coefficient not in `coef_bounds`.

    fit_intercept : bool, default=True
        Whether to fit an intercept in the underlying ConstrainedRegression.

    minimize_kwargs : dict, optional
        Extra options passed to scipy.optimize.minimize.

    func : callable, optional
        Custom prediction function for ConstrainedRegression.

    param_names : list of str, optional
        Names for the coefficients passed to ConstrainedRegression.

    alphas : array-like, default=(0.0,)
        Grid of alpha values (regularization strength) to search.

    l1_ratios : array-like, default=(0.0,)
        Grid of l1_ratio values (balance between L1 and L2) to search.

    cv : int or cross-validation generator, default=5
        Determines the cross-validation splitting strategy.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel for GridSearchCV.

    verbose : int, default=0
        Verbosity level passed to GridSearchCV.

    scoring : str or callable, default=None
        Scoring metric for GridSearchCV (None uses estimator's default score).
        
    use_smart_init : bool, default=True
        Whether to use smart initialization strategy.
        
    use_warm_start : bool, default=True
        Whether to use warm starts between CV iterations.
    """
    def __init__(
        self,
        coef_bounds=None,
        default_bounds=(None, None),
        fit_intercept=True,
        minimize_kwargs=None,
        func=None,
        param_names=None,
        alphas=(0.0,),
        l1_ratios=(0.0,),
        cv=5,
        n_jobs=None,
        verbose=0,
        scoring=None,
        use_smart_init=True,
        use_warm_start=True,
    ):
        self.coef_bounds = coef_bounds
        self.default_bounds = default_bounds
        self.fit_intercept = fit_intercept
        self.minimize_kwargs = minimize_kwargs
        self.func = func
        self.param_names = param_names
        self.alphas = alphas
        self.l1_ratios = l1_ratios
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.scoring = scoring
        self.use_smart_init = use_smart_init
        self.use_warm_start = use_warm_start

    def _create_param_grid(self):
        """Create optimized parameter grid with logarithmic spacing if needed."""
        # Use logarithmic spacing if alphas span more than 2 orders of magnitude
        alpha_grid = self.alphas
        if len(alpha_grid) > 3 and max(alpha_grid) / (min(alpha_grid) + 1e-10) > 100:
            # Generate logarithmic grid instead
            alpha_grid = np.logspace(
                np.log10(max(1e-10, min(alpha_grid))),
                np.log10(max(alpha_grid)),
                len(alpha_grid)
            )
        
        return {
            "alpha": alpha_grid,
            "l1_ratio": self.l1_ratios
        }
        
    def _get_initial_estimator(self, X, y):
        """Create and fit a quick initial estimator for warm starting."""
        base_est = ConstrainedRegression(
            coef_bounds=self.coef_bounds,
            default_bounds=self.default_bounds,
            fit_intercept=self.fit_intercept,
            minimize_kwargs=self.minimize_kwargs or {"maxiter": 20},  # Quick fit
            func=self.func,
            param_names=self.param_names,
            use_smart_init=self.use_smart_init
        )
        # Fit with minimal regularization
        base_est.alpha = min(0.01, max(self.alphas)) if len(self.alphas) > 1 else self.alphas[0]
        base_est.l1_ratio = 0.5
        try:
            base_est.fit(X, y)
            return base_est
        except:
            # If quick fit fails, return None
            return None

    def fit(self, X, y):
        # Base estimator with default alpha=0.0, l1_ratio=0.0 (overridden in grid)
        base_est = ConstrainedRegression(
            coef_bounds=self.coef_bounds,
            default_bounds=self.default_bounds,
            fit_intercept=self.fit_intercept,
            minimize_kwargs=self.minimize_kwargs,
            func=self.func,
            param_names=self.param_names,
            use_smart_init=self.use_smart_init
        )
        
        # Define hyperparameter grid (potentially optimized)
        param_grid = self._create_param_grid()
        
        # For warm start, try to get initial coefficients
        initial_est = None
        if self.use_warm_start:
            initial_est = self._get_initial_estimator(X, y)
            if hasattr(initial_est, 'coef_'):
                # Attach warm start info to base estimator class
                base_est._warm_start_coef = initial_est.coef_
                if hasattr(initial_est, 'intercept_'):
                    base_est._warm_start_intercept = initial_est.intercept_
                # Custom init method for warm start that monkey patches GridSearchCV
                def init_warm_start(estimator, **params):
                    estimator.set_params(**params)
                    if hasattr(base_est, '_warm_start_coef'):
                        estimator._warm_start_coef = base_est._warm_start_coef
                        if hasattr(base_est, '_warm_start_intercept'):
                            estimator._warm_start_intercept = base_est._warm_start_intercept
                    return estimator
                
                # Patch ConstrainedRegression to use warm start
                orig_fit = base_est.fit
                def warm_start_fit(self, X, y):
                    # If we have warm start coefs, use them
                    if hasattr(self, '_warm_start_coef'):
                        if self.use_smart_init and not hasattr(self, '_used_warm_start'):
                            n_params = len(self._warm_start_coef)
                            if self.fit_intercept:
                                n_params += 1
                            self._compute_smart_init = lambda X, y: np.concatenate(
                                [[self._warm_start_intercept], self._warm_start_coef]
                            ) if self.fit_intercept else self._warm_start_coef
                            # Mark used so we don't reuse if fit is called again
                            self._used_warm_start = True
                    return orig_fit(self, X, y)
                
                # Apply the monkey patch only for this instance
                base_est.fit = types.MethodType(warm_start_fit, base_est)
        
        # Perform grid search
        self.grid_ = GridSearchCV(
            base_est,
            param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            refit=True
        )
        self.grid_.fit(X, y)

        # Store best results
        self.best_estimator_ = self.grid_.best_estimator_
        self.best_params_    = self.grid_.best_params_
        self.best_alpha_     = self.best_params_["alpha"]
        self.best_l1_ratio_  = self.best_params_["l1_ratio"]
        self.best_score_     = self.grid_.best_score_

        # Mirror attributes
        self.coef_       = self.best_estimator_.coef_
        self.intercept_  = self.best_estimator_.intercept_
        self.coef_dict_  = self.best_estimator_.coef_dict_

        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)




# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_regression

    # 1) Linear Regression with non-negative x0
    X, y = make_regression(100, 2, noise=10, random_state=42)
    lr = ConstrainedRegression(
        coef_bounds={"x0": (0, None), "x1": (None, None)},
        param_names=["x0", "x1"],
        fit_intercept=True
    ).fit(X, y)
    print("Linear Example:")
    lr.summary()

    # 2) Nonlinear Cost-Improvement Curve
    # Create a dataframe with unit number and rate quantity
    # representing lot average cost and midpoint quantity
    # where the first 1/3 of the units have a lower rate quantity
    # and the last 1/3 have a higher rate quantity
    # and the middle 1/3 have a normal rate quantity
    df = pd.DataFrame({
        "unit_number": np.arange(1, 51),
        "rate_quantity": np.random.uniform(1, 10, 50)
    })
    true_T1, true_b, true_c = 100, -0.3, -0.1
    df["cost"] = true_T1 * df.unit_number**true_b * df.rate_quantity**true_c
    df["cost"] += np.random.normal(scale=2.0, size=50)

    def cost_curve(X, coefs):
        return coefs.T1 * X.unit_number**coefs.b * X.rate_quantity**coefs.c

    cic = ConstrainedRegression(
        func=cost_curve,
        param_names=["T1", "b", "c"],
        coef_bounds={"b": (-1, 0), "c": (-1, 0)},
        fit_intercept=False,
        default_bounds=(1e-3, None),
        minimize_kwargs={"method": "L-BFGS-B"}
    ).fit(df[["unit_number","rate_quantity"]], df["cost"])
    print("\nCost Improvement Curve Example:")
    cic.summary()


    # generate fake data
    #np.random.seed(1729)
    xdata = pd.DataFrame({'midpoint': np.arange(1,11), 'QTY': np.random.uniform(1,30,size=10), 'shutdown': np.random.uniform(0,1,size=10)})
    y = OLD_LearnCurve.func(xdata, 100, .95, .83)*(1.2**xdata.shutdown)
    
    y_noise =  np.random.normal(1, .10, size=xdata.shape[0])
    ydata = y * y_noise
    
    LC = LearnCurve()
    LC.fit(xdata,ydata)
    LC.predict(xdata)
    print("\nLearning Curve Example:")
    #print(LC.model)

    # compare to constrained model
    def lc_curve(X, coefs):
        
        b= np.log(coefs.LC) / np.log(2)
        r= np.log(coefs.RC) / np.log(2)
        ans = coefs.T1*(X.midpoint **b) * (X.QTY**r)*(coefs.shutdown_factor**X.shutdown)
        return ans
    model = ConstrainedRegression(
        func=lc_curve,
        param_names=["T1", "LC", "RC", "shutdown_factor"],
        coef_bounds={"LC": (.7, 1), "RC": (.7, 1), "shutdown_factor": (1,5)},
        fit_intercept=False,
        l1_ratio=0.2,
        alpha=0.02,
        minimize_kwargs={"method": "L-BFGS-B"}
    )
    model.fit(xdata,ydata)
    print("\nConstrained Learning Curve Example:")
    model.summary()

    from sklearn.model_selection import GridSearchCV
    # Generate data
    X, y = make_regression(200, 3, noise=5, random_state=0)
    df = pd.DataFrame(X, columns=["f1", "f2", "f3"])

    # 1) Instantiate your base estimator
    base = ConstrainedRegression(
        coef_bounds={"x0": (0,None)}, 
        param_names=["x0","x1","x2"]
    )

    # 2) Define the grid on **its** parameters
    grid = {"alpha": [0, 0.1], "l1_ratio":[0,0.5,1]}

    # 3) Create & fit GridSearchCV
    gscv = GridSearchCV(base, grid, cv=3)
    gscv.fit(X, y)

    print("Done without recursion!")

    ## Dont Run this it's not working
    if True:
        # Generate data
        X, y = make_regression(200, 3, noise=5, random_state=0)
        df = pd.DataFrame(X, columns=["f1", "f2", "f3"])

        # Hyperparameter grid
        alphas = [0.0, 0.1, 1.0]
        l1_ratios = [0.0, 0.5, 1.0]

        # Cross-validated constrained regression
        cr_cv = ConstrainedRegressionCV(
            coef_bounds={"f1": (0, None)},      # constrain f1 >= 0
            default_bounds=(None, None),
            fit_intercept=True,
            func=None,
            param_names=["f1", "f2", "f3"],
            alphas=alphas,
            l1_ratios=l1_ratios,
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring=None #"r2"
        )
        cr_cv.fit(df, y)

        print("Best alpha:", cr_cv.best_alpha_)
        print("Best l1_ratio:", cr_cv.best_l1_ratio_)
        print("Best CV score:", cr_cv.best_score_)
        cr_cv.best_estimator_.summary()

        # Calculate run times between smart and non-smart init
        from time import time
        t0 = time()
        cr_cv.fit(df, y)
        t1 = time()
        print(f"Smart init time taken: {t1-t0} seconds") 
        print(cr_cv.best_estimator_.summary())
        t0 = time()
        cr_cv.use_smart_init = False
        cr_cv.fit(df, y)
        t1 = time()
        print(f"Non-smart init time taken: {t1-t0} seconds")      
        print(cr_cv.best_estimator_.summary())

        # Calculate run without CV
        t0 = time()
        model = ConstrainedRegression(
            coef_bounds={"f1": (0, None)},      # constrain f1 >= 0
            default_bounds=(None, None),
            fit_intercept=True,
            func=None,
            param_names=["f1", "f2", "f3"],
            use_smart_init=False
        )
        model.fit(df, y)
        t1 = time()
        print(f"Non-CV time taken: {t1-t0} seconds")
        print(model.summary())

        # calculate run time for Ridge
        from sklearn.linear_model import Ridge
        t0 = time()
        ridge = Ridge(alpha=1)
        ridge.fit(df, y)
        t1 = time()
        print(f"Ridge time taken: {t1-t0} seconds")
        print(ridge.coef_)

        # calculate run time for LinearRegression
        from sklearn.linear_model import LinearRegression
        t0 = time()
        linreg = LinearRegression()
        linreg.fit(df, y)
        t1 = time()
        print(f"LinearRegression time taken: {t1-t0} seconds")
        print(linreg.coef_)