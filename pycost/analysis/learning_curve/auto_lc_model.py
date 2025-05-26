'''
Auto Learning Curve Model Module

This module provides the AutoLearningCurveModel class, which automates the process of selecting
and tuning the best learning curve model for a given dataset using cross-validation.

The class handles feature preparation, model selection, and hyperparameter tuning to find
the optimal learning curve model configuration.
'''

# Data manipulation
import numpy as np
import pandas as pd

# sklearn
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, KFold

# Import from our module
from pycost.analysis.learning_curve.lc_model import LearningCurveRegressor, r_squared


class AutoLearningCurveModel:
    """
    Automated model selection and tuning for learning curve regression.
    
    This class automatically selects and tunes the best learning curve model 
    using cross-validation. It handles feature engineering, model selection,
    and hyperparameter optimization.
    
    Parameters
    ----------
    auc_col : str
        Column name for the target variable (average unit cost).
    unit_col : str
        Column name for the unit number (cumulative production).
    rate_col : str
        Column name for the rate quantity (production rate).
    dummy_cols : list of str, optional
        Column names for categorical variables to be one-hot encoded.
    indep_cols : list of str, optional
        Column names for additional independent variables.
        
    Examples
    --------
    >>> auto_model = AutoLearningCurveModel(
    ...     auc_col='unit_cost', 
    ...     unit_col='unit_number', 
    ...     rate_col='rate_quantity',
    ...     dummy_cols=['program']
    ... )
    >>> auto_model.fit(df)
    >>> predictions = auto_model.predict(new_data)
    """
    def __init__(self, auc_col, unit_col, rate_col, dummy_cols=None, indep_cols=None):
        """
        Initialize the AutoLearningCurveModel with column specifications.
        
        Parameters
        ----------
        auc_col : str
            Column name for the target variable (average unit cost).
        unit_col : str
            Column name for the unit number (cumulative production).
        rate_col : str
            Column name for the rate quantity (production rate).
        dummy_cols : list of str or str, optional
            Column names for categorical variables to be one-hot encoded.
        indep_cols : list of str or str, optional
            Column names for additional independent variables.
        """
        self.auc_col = auc_col
        self.unit_col = unit_col
        self.rate_col = rate_col
        
        # Handle inputs that might be strings instead of lists
        if dummy_cols is not None:
            if isinstance(dummy_cols, str):
                self.dummy_cols = [dummy_cols]
            else:
                self.dummy_cols = dummy_cols
        else:
            self.dummy_cols = []
            
        if indep_cols is not None:
            if isinstance(indep_cols, str):
                self.indep_cols = [indep_cols]
            else:
                self.indep_cols = indep_cols
        else:
            self.indep_cols = []

    def _prepare_pipeline(self):
        """
        Prepare the preprocessing and model pipeline.
        
        Creates a sklearn pipeline that:
        1. Standardizes numerical features
        2. One-hot encodes categorical features
        3. Applies the LearningCurveRegressor
        
        Returns
        -------
        sklearn.pipeline.Pipeline
            The prepared pipeline ready for training.
        """
        # Define preprocessing for numerical features
        num_features = ['log_unit_number', 'log_rate_quantity'] + self.indep_cols
        num_transformer = StandardScaler()

        # Define preprocessing for categorical features
        cat_features = self.dummy_cols
        cat_transformer = OneHotEncoder(handle_unknown='ignore')

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_features),
                ('cat', cat_transformer, cat_features)
            ])

        # Create the pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LearningCurveRegressor(auc_col=self.auc_col, unit_col=self.unit_col, rate_col=self.rate_col))
        ])

        return pipeline

    def fit(self, df, model_type='ols', cv_folds=5, param_grid=None):
        """
        Fit the auto learning curve model to the data.
        
        Performs a grid search over model types and hyperparameters using cross-validation
        to find the optimal learning curve model.
        
        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe containing the learning curve data.
        model_type : str, default='ols'
            The type of model to use. Currently not used as grid search covers multiple models.
        cv_folds : int, default=5
            Number of cross-validation folds.
        param_grid : dict, optional
            Parameter grid for grid search. If None, a default grid is used.
            
        Returns
        -------
        self : AutoLearningCurveModel
            The fitted model.
            
        Notes
        -----
        The model selection is based on minimizing mean squared error through cross-validation.
        """
        # Create a copy to avoid modifying the original DataFrame
        df = df.copy()
        self.df_columns = df.columns.tolist()
        
        # Feature engineering: log-transform the specified columns
        df['log_unit_number'] = np.log(df[self.unit_col])
        df['log_rate_quantity'] = np.log(df[self.rate_col])

        # Define features and target
        feature_cols = ['log_unit_number', 'log_rate_quantity'] + self.dummy_cols + self.indep_cols
        X = df[feature_cols]
        y = df[self.auc_col]

        # Prepare the pipeline
        pipeline = self._prepare_pipeline()

        # Define default parameter grid if not provided
        if param_grid is None:
            # Default grid searches across different regression models and their hyperparameters
            param_grid = {
                'regressor__model': [LinearRegression(), LassoCV(), RidgeCV(), ElasticNetCV()],
                'regressor__alpha': [0.1, 1.0, 10.0],
                'regressor__l1_ratio': [0.1, 0.5, 0.9]
            }

        # Set up cross-validation
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Initialize GridSearchCV
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)

        # Store the best model and results
        self.best_model_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.cv_results_ = grid_search.cv_results_
        
        # Calculate and store R-squared on the training data
        y_pred = self.predict(df)
        self.r_squared_ = r_squared(y, y_pred)

        return self

    def predict(self, df):
        """
        Predict using the best model found during fitting.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the features for prediction.
            
        Returns
        -------
        numpy.ndarray
            Predicted values.
            
        Raises
        ------
        AttributeError
            If the model has not been fitted yet.
        """
        if not hasattr(self, 'best_model_'):
            raise AttributeError("Model has not been fitted yet. Call 'fit' first.")
            
        # Create a copy to avoid modifying the original DataFrame
        df = df.copy()

        # Log-transform the specified columns
        df['log_unit_number'] = np.log(df[self.unit_col])
        df['log_rate_quantity'] = np.log(df[self.rate_col])

        # Define features
        feature_cols = ['log_unit_number', 'log_rate_quantity'] + self.dummy_cols + self.indep_cols
        X = df[feature_cols]

        # Predict using the best model
        return self.best_model_.predict(X)

    def get_model_details(self):
        """
        Get details about the best model.
        
        Returns
        -------
        dict
            Dictionary containing details about the best model, including model type, 
            hyperparameters, and cross-validation scores.
            
        Raises
        ------
        AttributeError
            If the model has not been fitted yet.
        """
        if not hasattr(self, 'best_model_'):
            raise AttributeError("Model has not been fitted yet. Call 'fit' first.")
            
        # Extract the model type and hyperparameters
        regressor = self.best_model_.named_steps['regressor']
        model_type = type(regressor.model).__name__
        
        # Get the cross-validation results
        cv_results = {
            'mean_test_score': -self.cv_results_['mean_test_score'],  # Convert back from negative MSE
            'std_test_score': self.cv_results_['std_test_score']
        }
        
        # Return model details
        return {
            'model_type': model_type,
            'best_params': self.best_params_,
            'cv_results': cv_results
        }

    def r_squared(self, df=None, y_true=None):
        """
        Calculate the R-squared (coefficient of determination) for the model.
        
        Parameters
        ----------
        df : pandas.DataFrame, optional
            DataFrame containing features for prediction.
        y_true : array-like, optional
            Actual target values. If not provided, extracts from df using auc_col.
            
        Returns
        -------
        float
            R-squared value between 0 and 1, where 1 indicates perfect prediction.
            
        Raises
        ------
        ValueError
            If the model has not been fitted or if df is not provided.
        """
        if not hasattr(self, 'best_model_'):
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
            
        if df is None:
            raise ValueError("DataFrame must be provided.")
            
        # Extract target if not provided
        if y_true is None:
            y_true = df[self.auc_col]
        
        # Get predictions
        y_pred = self.predict(df)
        
        # Calculate R-squared
        return r_squared(y_true, y_pred)
    
    def score(self, df):
        """
        Return the coefficient of determination (R²) of the prediction.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing features and target.
            
        Returns
        -------
        float
            R-squared value between 0 and 1, where 1 indicates perfect prediction.
        """
        return self.r_squared(df, df[self.auc_col]) 