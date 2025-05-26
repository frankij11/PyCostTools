#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Analysis Example Script

This example demonstrates how to use the Model and Models classes
from pycost.analysis to perform regression analysis on cost data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler



# Import pycost modules
from pycost.analysis.model import Model, Models
from pycost.analysis.constrained.constrained_model import ConstrainedRegression, ConstrainedRegressionCV
# ======================================================
# Create a sample dataset for demonstration
# ======================================================
print("Generating sample dataset...")

np.random.seed(42)  # For reproducibility

# Generate sample data - simulating project cost factors
n_samples = 100
data = {
    'weight': np.random.uniform(10, 1000, n_samples),  # pounds
    'speed': np.random.uniform(20, 200, n_samples),    # mph
    'complexity': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'fuel_type': np.random.choice(['Gasoline', 'Diesel', 'Electric'], n_samples),
    'year': np.random.randint(2015, 2024, n_samples)
}

df = pd.DataFrame(data)

# Generate the target variable (total_cost) with some noise
# total_cost depends on project_size, material_cost, labor_hours, and complexity
complexity_factor = {'Low': 1.0, 'Medium': 1.3, 'High': 1.8}
df['complexity_factor'] = df['complexity'].map(complexity_factor)
fuel_type_factor = {'Gasoline': 1.0, 'Diesel': 1.2, 'Electric': 0.8}
df['fuel_type_factor'] = df['fuel_type'].map(fuel_type_factor)
df['total_cost'] = (
    df['weight'] * np.random.uniform(0.5, 1.5) +
    df['speed'] * np.random.uniform(35, 55) +
    df['year'] * np.random.uniform(1, 1.035) # Escalation rate
     + np.random.normal(50, 85) # Intercept
) * (1 + np.random.normal(0, 0.1, n_samples))  # Add 10% noise

df['total_cost'] = df['total_cost'] * df['complexity_factor'] * df['fuel_type_factor']

true_formula = "total_cost ~ (weight + speed + year):complexity:fuel_type"


print("Sample dataset created:")
print(df.head())

# ======================================================
# Single Model Analysis
# ======================================================
print("\n" + "="*50)
print("Single Model Analysis")
print("="*50)

# Create and fit a Model
print("\nCreating and fitting a single model...")
model = Model(
    df=df,
    formula="total_cost ~ weight + speed + year+ complexity + fuel_type",
    model=LinearRegression(),
    test_split=0.2,
    random_state=42,
    title="Project Cost Analysis",
    description="Predicting total project cost based on various factors",
    analyst="Data Scientist"
)

# Fit the model
model.fit()

# Print model summary
print("\nModel Summary:")
print(model.summary())

# Make predictions using the model
print("\nPredictions for the first 5 projects:")
predictions = model.predict(df.head())
print(predictions)

# Get model parameters
print("\nModel Coefficients:")
try:
    print(f"Intercept: {model.model['model'].intercept_}")
    print(f"Coefficients: {model.model['model'].coef_}")
except:
    print("Could not extract coefficients for this model type")


print("\nGenerating interactive report...")
model.fit()
report = model.report(show=False)

# Save the report as HTML
print("Saving report as HTML...")
report.save("project_cost_analysis_report.html")

print("Report saved! Open the HTML file to view the interactive report.")



# ======================================================
# Saving and Loading Models
# ======================================================
print("\n" + "="*50)
print("Saving and Loading Models")
print("="*50)

# Save the best model
print("\nSaving the best model...")
model.save("best_cost_model")

# Load the model back
print("\nLoading the saved model...")
loaded_model = Model.load("best_cost_model.joblib")
print("Model loaded successfully!")


model2 = Model(
    df=df,
    formula="total_cost ~ np.log(weight) + np.log(speed) + np.log(year) + complexity + fuel_type",
    model=LinearRegression(),
    test_split=0.2,
    random_state=42,
    title="Project Cost Analysis",
    y_transform=np.log,
    y_inverse=np.exp,
    description="Predicting total project cost based on various factors",
    analyst="Data Scientist"
)

model2.fit()
report2 = model2.report(show=False)
report2.save("report2.html")



# ======================================================
# Multiple Models Comparison
# ======================================================
print("\n" + "="*50)
print("Multiple Models Comparison")
print("="*50)

# Create different formulas to test
formulas = [
    "total_cost ~ weight + speed",
    "total_cost ~ weight + speed + year",
    "total_cost ~ weight + speed + year + complexity",
    "total_cost ~ weight + speed + year + complexity + fuel_type",
    true_formula
]

# Create different models to compare
models_to_test = [
    LinearRegression(),
    RidgeCV(alphas=np.logspace(-3, 3, 10)),
    LassoCV(alphas=np.logspace(-3, 3, 10)),
    RandomForestRegressor(n_estimators=100, random_state=42),
    ConstrainedRegression( default_bounds=(0, None), l1_ratio=0.01, alpha=0.01), # add a tiny amount of ridge regression to the lasso
    ConstrainedRegressionCV( default_bounds=(0, None), l1_ratios=[0.01, 0.1, 0.5, 1.0], alphas=[0.01, 0.1, 1.0])
]



# Create a Models instance to compare different model combinations
print("\nComparing multiple models with different formulas...")
models = Models(
    df=df,
    formulas=formulas,
    models=models_to_test,
    test_split=0.2,
    random_state=42,
    title="Project Cost Model Comparison",
    description="Comparing different model types and formulas",
    analyst="Data Scientist"
)

# Fit all models
print("\nFitting all models...")
models.fit(timeout_in_seconds=60)

# Get summary of all models
print("\nSummary of all models:")
summary = models.summary()
print(summary[['Formula', 'Model', 'TestRSQ', 'TestMSE', 'TestAbsErr']].sort_values('TestRSQ', ascending=False))

# ======================================================
# Create a HTML Report of Models Comparison
# ======================================================
reports = models.report(show=False)
reports.save("models_comparison_report.html")


# ======================================================
# Visualization and Reporting (if holoviews/panel are available)
# ======================================================
print("\n" + "="*50)
print("Visualization and Reporting")
print("="*50)

# ======================================================
# Create a model for each category of fuel type
# ======================================================
print("\n" + "="*50)
print("Creating a model for each category of fuel type")
print("="*50)

# Create a model for each category of fuel type
fuel_models = Models(df, formulas="total_cost ~ weight + speed + year + complexity", by="fuel_type", models=models_to_test)
fuel_models.fit()

# Get summary of all models
print("\nSummary of all models:")
summary = fuel_models.summary()
#print(summary[['Formula', 'Model', 'TestRSQ', 'TestMSE', 'TestAbsErr']].sort_values('TestRSQ', ascending=False))


reports = fuel_models.report(show=False)
reports.save("fuel_models_comparison_report.html")





