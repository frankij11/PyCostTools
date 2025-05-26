"""
PyCost Analysis Module

This module provides classes and functions for various types of cost analysis,
including standard regression models, learning curve models, and constrained
regression models.
"""

# Import from main analysis module
from pycost.analysis.model import (
    Model,
    LC_Model,
    Models,
    LC_Models,
)

from pycost.analysis.process import (
    AutoPreProcess,
    DateTransform,
    FeatureCheck,
    ImputeNA,
    MakeFormula,
    Clean,
    setup_logging
)

from pycost.analysis.auto_model import (
    AutoPipeline,
    AutoRegressionTrees,
    AutoRegressionLinear
)

# Import from constrained submodule
from pycost.analysis.constrained.constrained_model import (
    ConstrainedRegression,
    ConstrainedRegressionCV,
    LearnCurve
)

from pycost.analysis.constrained.simulation import (
    generate_multicollinear_data,
    monte_carlo_simulation,
    plot_simulation_results,
    run_monte_carlo_study
)

# Import from learning_curve submodule
from pycost.analysis.learning_curve.lc_model import (
    LearningCurveRegressor,
    ConstrainedLearningCurveModel,
    r_squared,
    plot_actual_vs_predicted
)

from pycost.analysis.learning_curve.auto_lc_model import (
    AutoLearningCurveModel
)

# Define module exports
__all__ = [
    # From model.py
    "Model", "LC_Model", "Models", "LC_Models", "AutoRegressionTrees", "AutoRegressionLinear",
    
    # From process.py
    "AutoPreProcess", "DateTransform", "FeatureCheck", "ImputeNA", "MakeFormula", "Clean", "setup_logging",
    
    # From auto_model.py
    "AutoPipeline",
    
    # From constrained_model.py
    "ConstrainedRegression", "ConstrainedRegressionCV", "LearnCurve",
    
    # From simulation.py
    "generate_multicollinear_data", "monte_carlo_simulation", "plot_simulation_results", "run_monte_carlo_study",
    
    # From lc_model.py
    "LearningCurveRegressor", "ConstrainedLearningCurveModel", "r_squared", "plot_actual_vs_predicted",
    
    # From auto_lc_model.py
    "AutoLearningCurveModel"
]
