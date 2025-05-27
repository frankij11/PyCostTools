"""
Core components for cost estimation.

This module provides the fundamental classes for the cost estimation framework:
- GlobalInputs: Program-wide parameters and settings
- Model: Base class for all cost models
- ParentModel: Container for multiple cost models
"""

from .base import GlobalInputs, Model
from .inventory import Inventory
from .parent_models import ParentModel

