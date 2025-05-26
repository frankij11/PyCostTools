"""
PyCost - Cost estimation framework for program analysis.

This package provides a framework for developing cost estimates using Python,
offering an alternative to traditional Excel-based cost estimation. It implements
a reactive programming model that automatically updates calculations when inputs change.

Main Components:
- GlobalInputs: Program-wide parameters and settings
- Model: Base class for cost estimation models
- ParentModel: Container for combining multiple cost models
- Specialized models (Development, Production, LearningCurve, Factor, EVM)
- UI components (ModelApp)
"""

# Version information
__version__ = "0.1.0"
__author__ = "PyCost Team"
