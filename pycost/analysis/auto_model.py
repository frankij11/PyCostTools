
# import necessary libraries
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin





class AutoPipeline:
    def __init__(
            self, df=None, formula=None, target=None, test_split=.2, random_state=42,
            handle_na=True, na_processor=None, preprocessor=None,
            meta_data=dict(title="My Report", desc="N/A", analyst="N/A"),
            scoring_function='neg_mean_squared_error', n_iter=50,
            **kwargs) -> None:

        args = locals()
        del args['self']
        del args['kwargs']
        for arg in args:
            self.__setattr__(arg, args[arg])

    @staticmethod
    def build_pipeline(X_train):
        categorical_values = []

        cat_subset = X_train.select_dtypes(
            include=['object', 'category', 'bool'])

        for i in range(cat_subset.shape[1]):
            categorical_values.append(
                list(cat_subset.iloc[:, i].dropna().unique()))

        date_pipeline = Pipeline([
            ('dateFeatures', process.DateTransform())
        ])

        num_pipeline = Pipeline([
            ('cleaner', SimpleImputer()),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('cleaner', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse=False, categories=categorical_values))
        ])

        preprocessor = ColumnTransformer([
            ('numerical', num_pipeline, make_column_selector(
                dtype_exclude=['object', 'category', 'bool'])),
            ('categorical', cat_pipeline, make_column_selector(
                dtype_include=['object', 'category', 'bool']))
        ])

        return preprocessor

class AutoRegressionTrees(BaseEstimator,RegressorMixin):
    def __init__(self, scoring_function='neg_mean_squared_error', n_iter=50) -> None:
        self.scoring_function = scoring_function
        self.n_iter = n_iter

    def fit(self, X, y):
        X_train = X
        y_train = y

        categorical_values = []

        cat_subset = X_train.select_dtypes(
            include=['object', 'category', 'bool'])

        for i in range(cat_subset.shape[1]):
            categorical_values.append(
                list(cat_subset.iloc[:, i].dropna().unique()))

        date_pipeline = Pipeline([
            ('dateFeatures', process.DateTransform())
        ])

        num_pipeline = Pipeline([
            ('cleaner', SimpleImputer()),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('cleaner', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse=False, categories=categorical_values))
        ])

        preprocessor = ColumnTransformer([
            ('numerical', num_pipeline, make_column_selector(
                dtype_exclude=['object', 'category', 'bool'])),
            ('categorical', cat_pipeline, make_column_selector(
                dtype_include=['object', 'category', 'bool']))
        ])

        model_pipeline_steps = []
        # model_pipeline_steps.append(('dateFeatures',date_pipeline))
        model_pipeline_steps.append(('preprocessor', preprocessor))
        model_pipeline_steps.append(
            ('feature_selector', SelectKBest(f_regression, k='all')))
        model_pipeline_steps.append(('estimator', RandomForestRegressor()))
        model_pipeline = Pipeline(model_pipeline_steps)

        total_features = preprocessor.fit_transform(X_train).shape[1]

        optimization_grid = []

        # Random Forest
        optimization_grid.append({
            'preprocessor__numerical__scaler': [None],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, round(total_features/10))) + ['all'],
            'estimator': [RandomForestRegressor(random_state=0)],
            'estimator__n_estimators': np.arange(5, 500, 10),
            'estimator__criterion': ['mse', 'mae']
        })

        # Gradient boosting
        # optimization_grid.append({
        #     'preprocessor__numerical__scaler':[None],
        #     'preprocessor__numerical__cleaner__strategy':['mean','median'],
        #     'feature_selector__k': list(np.arange(1,total_features,5)) + ['all'],
        #     'estimator':[GradientBoostingClassifier(random_state=0)],
        #     'estimator__n_estimators':np.arange(5,500,10),
        #     'estimator__learning_rate':np.linspace(0.1,0.9,20),
        # })

        search = RandomizedSearchCV(
            model_pipeline,
            optimization_grid,
            n_iter=self.n_iter,
            scoring=self.scoring_function,
            n_jobs=-1,
            random_state=0,
            verbose=3,
            cv=5
        )

        search.fit(X_train, y_train)
        self.best_estimator_ = search.best_estimator_
        self.best_pipeline = search.best_params_

    def predict(self, X, y=None):
        return self.best_estimator_.predict(X)

    def save(self, fname, compress=3):

        pass

    def summary(self):
        return Model.stats(self.X_test, self.y_test, self.X_train, self.y_train)


class AutoRegressionLinear(BaseEstimator,RegressorMixin):
    def __init__(self, scoring_function='neg_mean_squared_error', n_iter=50):
        self.scoring_function = scoring_function
        self.n_iter = n_iter

        # Impute
        # Drop Columns
        # scale
        # Add Features
        # Feature Selection
        # Estimators
        # Grid Search, Random Search

    def fit(self, X, y):
        X_train = self.X_train =self.X_test = X 
        y_train  = self.y_train =self.y_test = y

        categorical_values = []

        cat_subset = X_train.select_dtypes(
            include=['object', 'category', 'bool'])

        for i in range(cat_subset.shape[1]):
            categorical_values.append(
                list(cat_subset.iloc[:, i].dropna().unique()))

        date_pipeline = Pipeline([
            ('dateFeatures', process.DateTransform())
        ])

        num_pipeline = Pipeline([
            ('cleaner', SimpleImputer()),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('cleaner', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse=False, categories=categorical_values))
        ])

        preprocessor = ColumnTransformer([
            ('numerical', num_pipeline, make_column_selector(
                dtype_exclude=['object', 'category', 'bool'])),
            ('categorical', cat_pipeline, make_column_selector(
                dtype_include=['object', 'category', 'bool']))
        ])

        model_pipeline_steps = []
        model_pipeline_steps.append(('CheckFeatures', process.FeatureCheck() ))
        model_pipeline_steps.append(('dateFeatures',date_pipeline))
        model_pipeline_steps.append(('preprocessor', preprocessor))
        #model_pipeline_steps.append(('pca', PCA(.9,svd_solver = 'full')))
        model_pipeline_steps.append(('feature_selector', SelectKBest(f_regression, k='all')))
        model_pipeline_steps.append(('estimator', LinearRegression()))
        model_pipeline = Pipeline(model_pipeline_steps)

        total_features = preprocessor.fit_transform(X_train).shape[1]
        feature_incr = max(1,round(total_features/10))

        optimization_grid = []

        # Linear Regression
        optimization_grid.append({
            'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            #'pca__n_components': np.arange(0.7,0.95,0.05),
            'feature_selector__k': list(np.arange(1, total_features, feature_incr)) + ['all'],
            'estimator': [LinearRegression()]
        })

        # Regularized Regression
        optimization_grid.append({
            'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            #'pca': ['passthrough'],
            'feature_selector__k': list(np.arange(1, total_features, feature_incr)) + ['all'],
            'estimator': [ElasticNetCV()],
            'estimator__l1_ratio': [0.01, .1, .5, .7, .9, .95, .99, 1],
            'estimator__n_alphas': [100]
        })
        search = RandomizedSearchCV(
            model_pipeline,
            optimization_grid,
            n_iter=self.n_iter,
            scoring=self.scoring_function,
            n_jobs=-1,
            random_state=0,
            verbose=3,
            cv=5
        )

        search.fit(X_train, y_train)
        self.cv_results_ = search.cv_results_
        self.best_score_ = search.best_score_
        self.refit_time_ = search.refit_time_
        self.best_estimator_ = search.best_estimator_
        self.best_pipeline = search.best_params_
        return self
    


    def predict(self, X, y=None):
        return self.best_estimator_.predict(X)

    def summary(self):
        return Model.stats(self.best_estimator_,self.X_test, self.y_test, self.X_train, self.y_train)


