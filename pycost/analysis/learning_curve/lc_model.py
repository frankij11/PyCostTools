'''
Learning Curve Modeling Module

This module provides tools for learning curve analysis and modeling in cost estimation:

1. LearningCurveRegressor: A flexible regression model for fitting learning curves with various model types
2. ConstrainedLearningCurveModel: A model for fitting learning curves with constraints on coefficients
3. Utility functions for model evaluation, visualization, and coefficient interpretation

Learning curves model how unit costs decrease as cumulative production increases, typically following 
a power law relationship. This module supports both unit learning (experience) and rate effects.
'''

# data manipulation
import numpy as np
import pandas as pd

# patsy
from patsy import dmatrices, build_design_matrices

# scipy
from scipy.optimize import minimize

# sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, KFold

# Plot the data
import matplotlib.pyplot as plt


class LearningCurveRegressor(BaseEstimator, RegressorMixin):
    '''
    A builder class for learning curve regression that can accept any sklearn-compatible model.
    
    This class generates the transformation of a dataframe to fit different learning curve models:
    - learn_only: Models cost reduction based only on cumulative production
    - rate_only: Models cost reduction based only on production rate
    - learn_and_rate: Models cost reduction based on both cumulative production and rate
    
    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        The regression model to use, must be compatible with sklearn.
    df : pandas.DataFrame, optional
        The dataframe to fit the model to.
    lc_type : str, default='learn_and_rate'
        The type of learning curve to fit, must be one of 'learn_only', 'rate_only', or 'learn_and_rate'.
    auc_col : str, default='unit_cost'
        The column name of the target variable (unit cost).
    unit_col : str, default='unit_number'
        The column name of the unit number (cumulative production).
    rate_col : str, default='rate_quantity'
        The column name of the rate quantity (production rate).
    dummy_cols : str or list of str, optional
        The column names of categorical variables to be encoded as dummy variables.
    indep_cols : str or list of str, optional
        The column names of additional independent variables.
    **kwargs : 
        Additional keyword arguments to pass to the model.
        
    Examples
    --------
    >>> df = pd.DataFrame({
    >>>     'unit_cost': [100, 90, 80, 75],
    >>>     'unit_number': [1, 2, 3, 4],
    >>>     'rate_quantity': [1, 1, 2, 2]
    >>> })
    >>> model = LearningCurveRegressor(df=df)
    >>> model.fit()
    >>> model.predict(df)
    '''
    def __init__(self, model=ElasticNetCV(l1_ratio=np.arange(0,.5,0.05),alphas=np.arange(0.01,1,0.1)),df=None,lc_type='learn_and_rate', auc_col='unit_cost', unit_col='unit_number', rate_col='rate_quantity',dummy_cols=None, indep_cols=None,**kwargs):
        '''
        Parameters:
        - model: The type of regression model to use, must be compatible with sklearn.
        - df: The dataframe to fit the model to.
        - lc_type: The type of learning curve to fit, must be one of 'learn_only', 'rate_only', or 'learn_and_rate'.
        - auc_col: The column name of the target variable.
        - unit_col: The column name of the unit number.
        - rate_col: The column name of the rate quantity.
        - dummy_cols: The column names of the dummy variables. Accepts a string or a list of strings.
        - indep_cols: The column names of the independent variables. Accepts a string or a list of strings.

        '''
        # validate that the model is compatible with sklearn
        if not isinstance(model, BaseEstimator):
            raise ValueError(f"Model must be compatible with sklearn, got {type(model)}")

        self.model = model
        self.df = df
        self.auc_col = auc_col
        self.unit_col = unit_col
        self.rate_col = rate_col
        self.lc_type = lc_type
        if lc_type not in ['learn_only', 'rate_only', 'learn_and_rate']:
            raise ValueError(f"lc_type must be one of 'learn_only', 'rate_only', or 'learn_and_rate', got {lc_type}")
        
        # Initialize the list to store additional variables for the formula
        self.formula_extra_vars = []
        
        # Process dummy columns for categorical variables
        if dummy_cols is not None:
            if type(dummy_cols) == str:
                # Convert single string to list with one element
                self.dummy_cols = [dummy_cols]
            elif type(dummy_cols) == list:
                self.dummy_cols = dummy_cols
            else:
                print("dummy_cols is not a list or string, skipping")
                pass
            # Add dummy variable terms to formula: C(var_name) format for Patsy
            self.formula_extra_vars.append(*["C("+col+")" for col in self.dummy_cols])
    
        # Process independent columns for numerical variables
        if indep_cols is not None:
            if type(indep_cols) == str:
                # Convert single string to list with one element
                self.indep_cols = [indep_cols]
            elif type(indep_cols) == list:
                self.indep_cols = indep_cols
            else:
                print("indep_cols is not a list or string, skipping")
                pass
            # Add independent variable terms directly to formula
            self.formula_extra_vars.append(*indep_cols)

        # Format the extra variables for the Patsy formula
        if not self.formula_extra_vars:
            self.formula_extra_vars = ""
        else:
            # Join with + for the formula string
            self.formula_extra_vars = " + " + " + ".join(self.formula_extra_vars)
        
        # Store additional kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Define the formula for the model based on the learning curve type
        if lc_type == 'learn_only':
            # Only include unit number (learning/experience effect)
            self.formula = f'np.log(unit_cost) ~ np.log(unit_number) {self.formula_extra_vars}-1'
        elif lc_type == 'rate_only':
            # Only include rate quantity (production rate effect)
            self.formula = f'np.log(unit_cost) ~ np.log(rate_quantity) {self.formula_extra_vars}-1'
        elif lc_type == 'learn_and_rate':
            # Include both unit number and rate quantity (learning and rate effects)
            self.formula = f'np.log(unit_cost) ~ np.log(unit_number) + np.log(rate_quantity) {self.formula_extra_vars}-1'
        
        print("Building Learning Curve Model with formula: ", self.formula)
        # define the formula for the unit space model (for reference)
        self.formula_unit_space = 'unit_cost ~ T1 * unit_number**b * np.log(rate_quantity)**c * C(dummy_var)**d + indep_var'

    def fit(self, X=None, y=None):
        """
        Fit the learning curve model to the data.
        
        Parameters
        ----------
        X : pandas.DataFrame, optional
            Features dataframe. If provided, y must also be provided.
        y : pandas.Series, optional
            Target variable. If provided, X must also be provided.
            
        Returns
        -------
        self : LearningCurveRegressor
            The fitted model.
            
        Notes
        -----
        If X and y are not provided, the dataframe passed during initialization will be used.
        This method transforms the data using patsy to create design matrices based on the formula.
        """
        # Use provided X, y if available, otherwise use the instance dataframe
        if y is not None and X is not None:
            df = pd.concat([X, y], axis=1)
        else:
            df = self.df

        # Use patsy to create the design matrices based on the formula
        # Returns log-transformed target (y) and design matrix for independent variables (X)
        y, X = dmatrices(self.formula, df, return_type='dataframe')
        # Store design info for prediction
        self.design_info_ = X.design_info
        
        # Fit the model to the transformed data
        self.model_ = self.model
        self.model_.fit(X, y)
        
        # calculate the R-squared value
        y_pred = self.predict(df)
        # Convert y to a numpy array of the same shape as y_pred for correct R-squared calculation
        y_true = y.values.ravel()
        self.r_squared_ = r_squared(y_true, y_pred)
        
        # Store feature names for interpretation
        self.feature_names_ = X.columns.tolist()
        
        # Try to get coefficients and intercept
        # Some sklearn models might not have these attributes
        try:
            self.coef_ = self.model_.coef_
            self.intercept_ = self.model_.intercept_
        except:
            print(type(self.model_),"Model does not have coef_ or intercept_")
        
        return self

    def predict(self, X):
        """
        Predict unit costs for new data.
        
        Parameters
        ----------
        X : pandas.DataFrame
            DataFrame containing features required by the formula.
            
        Returns
        -------
        np.ndarray
            Predicted unit costs (not log-transformed, in original scale).
        """
        # Transform the input data using the design matrix from fitting
        X_transformed = build_design_matrices([self.design_info_], X)[0]
        
        # Get log-space predictions
        y_log_pred = self.model_.predict(X_transformed)
        
        # Transform back to original scale (exponentiate)
        return np.exp(y_log_pred)

    def r_squared(self, X=None, y=None):
        """
        Calculate the R-squared (coefficient of determination) for the model.
        
        Parameters
        ----------
        X : pandas.DataFrame, optional
            Features dataframe. If not provided, uses the dataframe from fitting.
        y : pandas.Series, optional
            Actual target values. If not provided, uses the target column from the dataframe.
            
        Returns
        -------
        float
            R-squared value between 0 and 1, where 1 indicates perfect prediction.
            
        Raises
        ------
        ValueError
            If the model has not been fitted or if both X and y are not provided and 
            no dataframe was provided during initialization.
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
            
        if X is None and y is None:
            if self.df is None:
                raise ValueError("No data provided and no dataframe was set during initialization.")
            X = self.df
            y = self.df[self.auc_col]
        elif y is None:
            y = X[self.auc_col]
            
        # Get predictions
        y_pred = self.predict(X)
        
        # Calculate R-squared
        return r_squared(y, y_pred)
    
    def score(self, X, y=None):
        """
        Return the coefficient of determination (R²) of the prediction.
        
        This is the sklearn-compatible scoring method that allows the model
        to be used with cross-validation and other sklearn tools.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Features dataframe.
        y : pandas.Series, optional
            Actual target values. If not provided, uses the target column from X.
            
        Returns
        -------
        float
            R-squared value between 0 and 1, where 1 indicates perfect prediction.
        """
        if y is None:
            y = X[self.auc_col]
            
        return self.r_squared(X, y)

class ConstrainedLearningCurveModel:
    """
    A learning curve model with constraints on coefficient values.
    
    This model allows for imposing bounds on coefficients to ensure they
    correspond to realistic learning effects (e.g., learning rates between 60% and 100%).
    
    Parameters
    ----------
    formula : str
        A Patsy formula string, e.g., 'avg_unit_cost ~ np.log(unit_number) + np.log(rate_quantity)'
    bounds : dict, optional
        Dictionary mapping feature names to (min, max) tuples for coefficient constraints.
    unit_number_col : str, default='unit_number'
        Column name for cumulative production units.
    rate_quantity_col : str, default='rate_quantity'
        Column name for production rate.
        
    Examples
    --------
    >>> formula = 'unit_cost ~ np.log(unit_number) + np.log(rate_quantity)'
    >>> bounds = {'np.log(unit_number)': (-0.8, 0), 'np.log(rate_quantity)': (-0.8, 0)}
    >>> model = ConstrainedLearningCurveModel(formula=formula, bounds=bounds)
    >>> model.fit(df)
    >>> predictions = model.predict(new_data)
    """
    def __init__(self, formula, bounds=None, unit_number_col='unit_number', rate_quantity_col='rate_quantity'):
        """
        Parameters:
        - formula: A Patsy formula string, e.g., 'avg_unit_cost ~ np.log(unit_number) + np.log(rate_quantity) + C(dummy_var) + indep_var'
        - bounds: Dictionary mapping feature names to (min, max) tuples.
        - unit_number_col: Column name for the unit number variable (default: 'unit_number')
        - rate_quantity_col: Column name for the rate quantity variable (default: 'rate_quantity')
        """
        self.formula = formula
        self.bounds = bounds
        self.coef_ = None
        self.feature_names_ = None
        self.unit_space_equation_ = None
        self.design_info_ = None  # To store design_info
        self.unit_number_col = unit_number_col
        self.rate_quantity_col = rate_quantity_col

    def fit(self, df):
        """
        Fit the constrained learning curve model to the data.
        
        Uses constrained optimization to find coefficients within the specified bounds
        that minimize the mean squared error.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing all variables referenced in the formula.
            
        Returns
        -------
        self : ConstrainedLearningCurveModel
            The fitted model.
        
        Raises
        ------
        ValueError
            If optimization fails.
        """
        # Parse the formula to get design matrices using Patsy
        y, X = dmatrices(self.formula, df, return_type='dataframe')
        self.design_info_ = X.design_info  # Save design_info for prediction
        self.feature_names_ = X.columns.tolist()
        
        # Store original target for R-squared calculation
        original_y = y.values.ravel()
        
        # Check if target is log-transformed in the formula
        target_is_log = self.formula.split('~')[0].strip().startswith('np.log(')
        
        # Handle log transformation of target if not already transformed in formula
        if not target_is_log:
            y = np.log(y.values.ravel())  # Log-transform the target variable
        else:
            y = y.values.ravel()

        # Define default bounds for coefficients
        # These defaults represent typical ranges for learning and rate effects
        default_bounds = {}
        for feature in self.feature_names_:
            if self.unit_number_col in feature:
                # Bounds for unit learning: 60% to 100% learning curve
                # Convert from learning curve percentage to log-space coefficient
                lower = np.log(0.6) / np.log(2)  # ~-0.74 for 60% learning curve
                upper = np.log(1.0) / np.log(2)  # 0 for 100% (no learning)
                default_bounds[feature] = (lower, upper)
            elif self.rate_quantity_col in feature:
                # Bounds for rate effects: 60% to 100% rate curve
                lower = np.log(0.6) / np.log(2)
                upper = np.log(1.0) / np.log(2)
                default_bounds[feature] = (lower, upper)
            else:
                # No constraints on other coefficients by default
                default_bounds[feature] = (0, None)

        # Override defaults with user-specified bounds if provided
        if self.bounds:
            for key, value in self.bounds.items():
                default_bounds[key] = value

        # Create bounds list in the order of feature_names_ for optimizer
        bounds_list = [default_bounds[feature] for feature in self.feature_names_]

        # Define the objective function (mean squared error)
        def objective(coefs):
            """Calculate mean squared error in log space"""
            return np.mean((y - X.values @ coefs) ** 2)

        # Initial guess for coefficients (start with zeros)
        x0 = np.zeros(X.shape[1])

        # Perform constrained optimization
        result = minimize(objective, x0, bounds=bounds_list)
        if not result.success:
            raise ValueError("Optimization failed: " + result.message)
        
        # Store optimized coefficients FIRST
        self.coef_ = result.x
        
        # THEN calculate the R-squared value
        y_pred = self.predict(df)
        
        # For R-squared calculation, we need to match the transformation state
        # If formula starts with log, predictions need to be logged to match original y
        # Otherwise, we use the raw predictions with the original target
        if target_is_log:
            self.r_squared_ = r_squared(y, np.log(y_pred))
        else:
            self.r_squared_ = r_squared(original_y, y_pred)

        # Generate human-readable equation in unit space
        # Convert log-space coefficients to power law form
        terms = []
        for coef, name in zip(self.coef_, self.feature_names_):
            if 'np.log(' in name:
                # Extract variable name from log() expression
                var_name = name.split('np.log(')[1].split(')')[0]
                # Convert to power law format
                terms.append(f"{var_name}**{coef:.4f}")
            elif name.lower() == 'intercept':
                # Exponentiate intercept for unit space
                terms.append(f"{np.exp(coef):.4f}")
            else:
                # Handle other terms
                terms.append(f"{np.exp(coef):.4f} * {name}")
        
        # Create the full unit space equation
        self.unit_space_equation_ = " * ".join(terms)

        return self

    def predict(self, df):
        """
        Predict unit costs for new data.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing all variables referenced in the formula.
            
        Returns
        -------
        np.ndarray
            Predicted unit costs in original scale.
        """
        # Generate design matrix for new data using saved design_info
        X_new = build_design_matrices([self.design_info_], df)[0]
        
        # Get log-space predictions
        y_pred_log = X_new @ self.coef_
        
        # Transform back to original scale
        return np.exp(y_pred_log)

    def r_squared(self, df=None, y_true=None):
        """
        Calculate the R-squared (coefficient of determination) for the model.
        
        Parameters
        ----------
        df : pandas.DataFrame, optional
            DataFrame containing features for prediction. If not provided and y_true is provided,
            uses the data from fitting.
        y_true : array-like, optional
            Actual target values. If not provided and df is provided, extracts from df
            based on the formula.
            
        Returns
        -------
        float
            R-squared value between 0 and 1, where 1 indicates perfect prediction.
            
        Raises
        ------
        ValueError
            If the model has not been fitted or if necessary data is not provided.
        """
        if not hasattr(self, 'coef_'):
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
            
        # Extract target from formula if needed
        if y_true is None and df is not None:
            target_var = self.formula.split('~')[0].strip()
            # Handle log transform if present
            if target_var.startswith('np.log('):
                target_var = target_var[7:-1]  # extract from np.log(...)
                y_true = np.log(df[target_var].values)
            else:
                y_true = df[target_var].values
        
        if df is None or y_true is None:
            raise ValueError("Either df or both df and y_true must be provided.")
            
        # Get predictions
        y_pred = self.predict(df)
        
        # If target is log-transformed in the formula, transform predictions back
        if self.formula.split('~')[0].strip().startswith('np.log('):
            y_pred = np.log(y_pred)
            
        # Calculate R-squared
        return r_squared(y_true, y_pred)
    
    def score(self, df):
        """
        Return the coefficient of determination (R²) of the prediction.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing all variables referenced in the formula.
            
        Returns
        -------
        float
            R-squared value between 0 and 1, where 1 indicates perfect prediction.
        """
        return self.r_squared(df)

def unstandardize_coefs(model, scaler):
    '''
    Convert the standardized coefficients to unstandardized coefficients.
    
    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        The fitted model with standardized coefficients.
    scaler : sklearn.preprocessing.StandardScaler
        The scaler used to standardize the features.
        
    Returns
    -------
    dict
        Dictionary of unstandardized coefficients with feature names as keys.
        
    Notes
    -----
    For a standardized model with y = β₀ + β₁z₁ + β₂z₂ + ... where z are standardized features,
    the unstandardized coefficients become:
    - β₀' = β₀ - Σ(βⱼ×μⱼ/σⱼ)
    - βⱼ' = βⱼ/σⱼ
    where μⱼ and σⱼ are the mean and standard deviation used for standardization.
    '''
    # Extract standardized coefficients
    standardized_coefs = model.coef_
    standardized_intercept = model.intercept_

    # Get means and standard deviations from the scaler
    means = scaler.mean_
    stds = scaler.scale_

    # Unstandardize the coefficients by dividing by the standard deviation
    unstandardized_coefs = standardized_coefs * stds
    
    # Adjust the intercept term
    unstandardized_intercept = standardized_intercept - np.sum(standardized_coefs * means)

    # Create a dictionary with feature names (if available) or indices
    if hasattr(model, 'feature_names_'):
        unstandardized_coefs = dict(zip(model.feature_names_, unstandardized_coefs))
        unstandardized_coefs['intercept'] = unstandardized_intercept
    else:
        unstandardized_coefs = dict(zip(range(len(unstandardized_coefs)), unstandardized_coefs))
        unstandardized_coefs['intercept'] = unstandardized_intercept

    return unstandardized_coefs

def r_squared(y_true, y_pred):
    """
    Calculate the coefficient of determination (R²).
    
    Parameters
    ----------
    y_true : array-like
        True values of the target variable.
    y_pred : array-like
        Predicted values of the target variable.
        
    Returns
    -------
    float
        R² value, between 0 and 1, where 1 indicates perfect prediction.
        
    Notes
    -----
    R² = 1 - SS_res / SS_tot
    where SS_res is the sum of squared residuals and SS_tot is the total sum of squares.
    """
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def plot_actual_vs_predicted(df, model, title='Actual vs Predicted', x_axis='unit_number', y_axis='unit_cost', group_col='group', coefs=None, fig=None):
    """
    Create a scatter plot comparing actual vs predicted values.

    If fig is provided, the plot will be added to the figure.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.
    model : object
        A fitted model with a predict method.
    title : str, default='Actual vs Predicted'
        Plot title.
    x_axis : str, default='unit_number'
        Column to use for x-axis.
    y_axis : str, default='unit_cost'
        Column containing actual y values.
    group_col : str, default='group'
        Column to use for grouping/coloring points.
    coefs : dict, optional
        Coefficients to display in the plot.
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        A figure with the actual vs predicted plot.
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Create the figure
    if fig is None:
        fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1,1,1)
    
    # Generate predictions and add to dataframe
    df['y_pred'] = model.predict(df)
    df['y_actual'] = df[y_axis]
    
    # Calculate R-squared for display
    r2 = r_squared(df['y_actual'], df['y_pred'])
    
    # Plot data with different colors based on group
    if group_col:
        # Create color codes based on group
        df[group_col+"_num"] = df[group_col].astype('category').cat.codes
        
        # Generate color maps - blues for actual values, reds for predictions
        # Make sure the colors are dark enough to be visible
        colors_actual = plt.cm.Blues(df[group_col+"_num"].astype(float)+1 / df[group_col+"_num"].nunique())
        colors_pred = plt.cm.Reds(df[group_col+"_num"].astype(float)+1 / df[group_col+"_num"].nunique())
        
        # Plot points
        ax.scatter(x=x_axis, y='y_actual', color=colors_actual, data=df)
        ax.scatter(x=x_axis, y='y_pred', color=colors_pred, data=df)
    else:
        # Use default colors if no grouping
        ax.scatter(x=x_axis, y='y_actual', color='blue', data=df)
        ax.scatter(x=x_axis, y='y_pred', color='red', data=df)
    
    # Add coefficients as text if provided
    if coefs is not None:
        # Create a text box for the coefficients
        text_box = f"Coefficients: {coefs}"
        ax.text(0.5, 0.5, text_box, ha='center', va='center', wrap=True)
    
    # Add title with R-squared value
    ax.set_title(title + f"\nR-squared: {r2:.4f}")
    
    # Add legend
    ax.legend(['Actual', 'Predicted'])
    
    # Add data table below the plot
    ax.table(cellText=df.head(10).values, colLabels=df.columns, loc='bottom')
    
    return ax

if __name__ == "__main__":
    # Sample DataFrame generation function
    def make_lc_data(n_groups=5, n_units=20, t1=100, lc_slope=0.95, rc_slope=0.95,n_indep_vars=0, t1_cv=.25, equation_cv=.25):
        '''
        Create a synthetic learning curve dataset with specified parameters.
        
        Parameters
        ----------
        n_groups : int, default=5
            Number of different groups/programs to generate.
        n_units : int, default=20
            Number of units per group.
        t1 : float, default=100
            Initial unit cost (theoretical first unit cost).
        lc_slope : float, default=0.95
            Learning curve slope (as a decimal, e.g., 0.95 for 95%).
        rc_slope : float, default=0.95
            Rate curve slope (as a decimal).
        indep_var : float, default=0
            Value for an independent variable.
        t1_cv : float, default=0.25
            Coefficient of variation for T1 (initial cost).
        equation_cv : float, default=0.25
            Coefficient of variation for the overall equation.
            
        Returns
        -------
        pandas.DataFrame
            Dataframe containing synthetic learning curve data.
        '''
        df = pd.DataFrame()
        for i in range(n_groups):
            # Create a ramp up and then steady state production profile
            # First 1/3 of units are in ramp-up phase, then steady state
            n_ramp_up = np.floor(n_units/3)
            n_steady_state = int(n_units - n_ramp_up)
            
            # Create ramp-up production quantities (increasing)
            ramp_up = np.arange(1, 1+n_ramp_up)
            
            # Create steady-state production quantities (constant)
            steady_state = np.array([1+n_ramp_up]*n_steady_state)
            
            # Combine into full production rate profile
            rate_quantity = np.concatenate([ramp_up, steady_state])
            
            # Generate group data
            group_data = pd.DataFrame({
                'group': ['program '+str(i)]*n_units,
                'FY': np.arange(2010, 2010+n_units),
                'unit_number': np.arange(1, 1+n_units),
                'rate_quantity': rate_quantity,
                'indep_var': np.arange(1, 1+n_units)
            })
            for i in range(n_indep_vars):
                group_data['indep_var_'+str(i)] = np.random.normal(np.random.uniform(15,40),np.random.uniform(1,5), size=n_units)
            

            # Apply learning curve formula with random variation
            # Cost = T1 * (unit_number^log(slope)/log(2)) * (rate_quantity^log(rate_slope)/log(2))
            group_data = group_data.assign(
                unit_cost = lambda x: np.random.normal(1, equation_cv, size=n_units) * 
                                      np.random.normal(t1, t1*t1_cv) * 
                                      (x.unit_number ** np.log(lc_slope)/np.log(2)) * 
                                      (x.rate_quantity ** np.log(rc_slope)/np.log(2))
            )
            # For each independent variable, apply a random slope
            # And update the unit_cost
            for i in range(n_indep_vars):
                group_data['unit_cost'] = group_data['unit_cost'] * (group_data['indep_var_'+str(i)] * np.random.uniform(0,1.5))
            
            # Add to main dataframe
            df = pd.concat([df, group_data])
        
        return df
    
    # Generate synthetic test data
    df = make_lc_data(n_groups=5, n_units=100, lc_slope=0.95, rc_slope=0.90, n_indep_vars=1, t1_cv=.25, equation_cv=.1)
    examples = {}

    model_linear = LearningCurveRegressor(
        df=df,
        model=LinearRegression(),
        auc_col='unit_cost',
        unit_col='unit_number',
        rate_col='rate_quantity',
        dummy_cols=['group'],
        indep_cols=['indep_var']
    )
    model_linear.fit()
    examples['linear'] = model_linear
    # plot the actual vs predicted
    plot_actual_vs_predicted(
        df, 
        model_linear, 
        title="Linear Model: " + model_linear.model.__class__.__name__ + '\n' + model_linear.formula, 
        coefs=model_linear.coef_
    )

    # Example 1: Unscaled ElasticNet model
    # Create a model with low L1 ratio (closer to Ridge) and low alphas
    enet_model = ElasticNetCV(l1_ratio=np.arange(0.0001, .5001, 0.05), alphas=np.arange(0.001, .2, 0.1))
    
    # Set up the model without scaling
    model_unscaled = LearningCurveRegressor(
        df=df,
        model=enet_model, 
        auc_col='unit_cost', 
        unit_col='unit_number', 
        rate_col='rate_quantity',
        dummy_cols=['group'], 
        indep_cols=['indep_var']
    )
    
    # Fit the model and print coefficients
    model_unscaled.fit() 
    coefs = dict(zip(model_unscaled.feature_names_, model_unscaled.coef_))
    for key, value in coefs.items():
        try:
            if 'np.log(' in key:
                print(f"{key}: {np.exp(value):.4f}")
            else:
                print(f"{key}: {value:.4f}")
        except:
            print(f"{key}: {value}")
    # plot the actual vs predicted
    plot_actual_vs_predicted(
        df, 
        model_unscaled, 
        title="Unscaled Model: " + model_unscaled.model.__class__.__name__ + '\n' + model_unscaled.formula, 
        coefs=coefs
    )
    examples['enet_unscaled'] = model_unscaled
    # Example 2: Scaled model with standardization
    # Set up a pipeline with standardization and ElasticNet
    scaler = StandardScaler()
    pipeline = Pipeline([('scaler', scaler), ('enet', enet_model)])
    
    # Create and fit the model with scaling
    model_scaled = LearningCurveRegressor(
        df=df,
        model=pipeline, 
        auc_col='unit_cost', 
        unit_col='unit_number', 
        rate_col='rate_quantity',
        dummy_cols=['group'], 
        indep_cols=['indep_var']
    )
    model_scaled.fit() 
    examples['enet_scaled'] = model_scaled
    # Extract and print unstandardized coefficients
    coefs = unstandardize_coefs(pipeline.named_steps['enet'], pipeline.named_steps['scaler'])
    for key, value in coefs.items():
        try:
            if 'np.log(' in key:
                print(f"{key}: {np.exp(value):.4f}")    
        except:
            print(f"{key}: {value}")
    # plot the actual vs predicted
    plot_actual_vs_predicted(
        df, 
        model_scaled, 
        title="Scaled Model: " + model_scaled.model.__class__.__name__ + '\n' + model_scaled.formula, 
        coefs=coefs
    )   

    # Example 3: Test different regression models
    alphas = np.arange(0.01,.2,0.1)
    l1_ratios = np.arange(0.01,.2,0.1)
    models = [LinearRegression(), 
              LassoCV(alphas=alphas), 
              RidgeCV(alphas=alphas), 
              ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas)]
    for model in models:
        model = LearningCurveRegressor(
            model=model,
            df=df, 
            auc_col='unit_cost', 
            unit_col='unit_number', 
            rate_col='rate_quantity',
            dummy_cols=['group'], 
            indep_cols=['indep_var']
        )
        model.fit()
        examples[model.model.__class__.__name__] = model

    
    # Plot actual vs predicted in the same figure
    """
    fig = plt.figure(figsize=(10, 6))
    for model in examples.values():
        ax = plot_actual_vs_predicted(
            df, 
            model, 
            title="Many Models: " + model.model.__class__.__name__ + '\n' + model.formula, 
            coefs=None,
            fig=fig
        )
    """
    # Example 4: Constrained model
    # Define formula for ConstrainedLearningCurveModel
    formula = 'unit_cost ~ np.log(unit_number) + np.log(rate_quantity) + group + indep_var'

    # Define coefficient bounds
    bounds = {
        'np.log(unit_number)': (np.log(0.6)/np.log(2), np.log(1.0)/np.log(2)),
        'np.log(rate_quantity)': (np.log(0.6)/np.log(2), np.log(1.0)/np.log(2)),
        'C(dummy_var)[T.B]': (0, None),
        'indep_var': (0, None)
    }

    # Initialize and fit the constrained model
    model = ConstrainedLearningCurveModel(formula=formula, bounds=bounds)
    model.fit(df)
    examples['constrained'] = model

    # Test predictions on new data
    new_data = pd.DataFrame({
        'unit_number': [5, 6],
        'rate_quantity': [50, 60],
        'group': ["program 0", "program 1"],
        'indep_var': [9, 10]
    })
    predictions = model.predict(new_data)
    print(predictions)

    # View the equation in unit space
    print("Unit Space Equation:")
    print(model.unit_space_equation_)
    
    # Plot actual vs predicted for constrained model
    plot_actual_vs_predicted(
        df, 
        model, 
        title=model.__class__.__name__ + '\n' + model.formula, 
        coefs=model.coef_
    )

    # Example 5: Simple model with large sample
    # Generate a larger, simpler dataset
    df2 = make_lc_data(n_groups=1, n_units=1000, t1=100, lc_slope=0.95, rc_slope=0.90, n_indep_vars=0, equation_cv=.1)
    
    # Define a simpler formula
    formula2 = 'unit_cost ~ np.log(unit_number) + np.log(rate_quantity)'
    
    # Create and fit the model
    model2 = ConstrainedLearningCurveModel(formula=formula2)
    model2.fit(df2)
    examples['constrained_simple'] = model2

    
    # Print coefficients
    coefs = dict(zip(model2.feature_names_, model2.coef_))
    for key, value in coefs.items():
        if 'np.log(' in key:
            print(f"{key}: {np.exp(value):.4f}")
        else:
            print(f"{key}: {np.exp(value):.4f}")

    # Plot the data for the simple model
    plot_actual_vs_predicted(
        df2, 
        model2, 
        title='Constrained Learning Curve\n' + model2.formula, 
        coefs=coefs
    )
    
    # print all the models R-squared values
    for key,model in examples.items():
        print(f"{key}: {model.r_squared_:.4f}")

    # print all the models coefficients
    for key,model in examples.items():
        try:
            print(f"{key}: {model.formula}")
            print(f"{key}: {model.coef_}") 
        except:
            print(f"{key}: No coefficients available")

    # Show all plots
    plt.show()

