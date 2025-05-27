# PyCostTools Development Tasks

This document outlines tasks to improve the PyCostTools package by fixing bugs, refactoring code, and implementing best practices.

## Code Structure and Organization

- [ ] [Medium] Reorganize package structure to follow modern Python conventions
  - [ ] [Medium] Implement proper `__init__.py` files for module exports
  - [ ] [Medium] Set up proper namespace hierarchy
  - [ ] [Medium] Create separate modules for related functionality

- [ ] [High] Documentation improvements
  - [ ] [Medium] Add type hints throughout the codebase
  - [ ] [High] Complete docstrings for all classes and methods
  - [ ] [Medium] Generate API documentation using Sphinx

## Bug Fixes

- [ ] [High] Fix inflation and escalation calculations (currently placeholders)
  - [ ] [High] Implement proper inflation conversion in `calc_cost_metadata`
  - [ ] [High] Add proper conversion between cost and calendar year dollars

- [ ] [High] Address issues in Monte Carlo simulation
  - [ ] [High] Fix `_end_sim` method to properly reset parameters
  - [ ] [High] Fix `calc_cost_estimate` reference (doesn't exist)
  - [ ] [High] Replace numbergen (ng) dependency with built-in SimEngine
      - [ ] [High] Update Model class to use SimEngine instance rather than global functions
      - [ ] [High] Implement seed management for reproducible simulations
      - [ ] [Medium] Add ability to create and manage multiple simulation engines
      - [ ] [Medium] Ensure consistent distribution parameter naming (mu/sigma vs mean/stdev)
      - [ ] [Medium] Update documentation and examples

- [ ] [High] Resolve DataFrame handling issues
  - [ ] [High] Fix column type/name inconsistencies (e.g., lowercase vs. uppercase column names)
  - [ ] [High] Ensure consistent column naming across models
  - [ ] [High] Fix usage of 'FY' vs 'fy' columns in scheduling calculations

- [ ] [High] Fix error handling and validation
  - [ ] [High] Add proper validation for required fields
  - [ ] [High] Add defensive checks before operations on potentially empty DataFrames

## Refactoring

- [ ] [High] Implement PEP 8 compliance
  - [ ] [High] Fix variable naming to follow snake_case convention
  - [ ] [Medium] Reduce line lengths to 79-88 characters
  - [ ] [Medium] Fix import order and grouping

- [ ] [Medium] Improve code modularity
  - [ ] [Medium] Extract common functionality into utility functions
  - [ ] [Medium] Decouple UI components from model logic
  - [ ] [Medium] Break up large classes into smaller, focused classes

- [ ] [Medium] Performance improvements
  - [ ] [Medium] Optimize DataFrame operations to reduce copy operations
  - [ ] [Medium] Improve parallelization in Monte Carlo simulation
  - [ ] [Medium] Use vectorized operations where possible

- [ ] [Medium] Simplify model interface
  - [ ] [Medium] Standardize parameter management across model types
  - [ ] [Medium] Create consistent API for all model types
  - [ ] [Medium] Implement interface templates/protocols for different model types

## Feature Implementation

- [ ] [Medium] Complete the Analysis module
  - [ ] [Medium] Implement regression analysis capabilities
  - [ ] [Medium] Add learning curve analysis functionalities
  - [ ] [Medium] Add optimization tools for parameter fitting

- [ ] [Medium] Enhance the CostModel module
  - [ ] [Medium] Add support for advanced WBS management
  - [ ] [Medium] Implement better uncertainty distribution models
  - [ ] [Low] Add time-phasing capabilities

- [ ] [Low] Add validation and quality assurance tools
  - [ ] [Low] Implement model validation checks
  - [ ] [Low] Add data quality assessment tools
  - [ ] [Low] Create model comparison utilities

- [ ] [Low] Create visualization and reporting tools
  - [ ] [Low] Standard cost report generators
  - [ ] [Low] Interactive dashboards for model analysis
  - [ ] [Low] Export capabilities for reports and presentations

## Testing

- [ ] [High] Set up comprehensive test suite
  - [ ] [High] Unit tests for core functionality
  - [ ] [Medium] Integration tests for model interactions
  - [ ] [Medium] Regression tests for known bugs

- [ ] [High] Set up CI/CD pipeline
  - [ ] [High] Automated testing on pushes/PRs
  - [ ] [High] Automated linting and style checking
  - [ ] [Medium] Package building and publishing workflows

## Release Preparation and Review
- [ ] [High] Finalize and verify `LICENSE.txt` details (e.g., copyright holder, year). (Self-note: This was mostly done, but good to keep as a checklist item for releases).
- [ ] [Medium] Ensure consistent logging practices throughout the library:
    - [ ] [Medium] Verify all modules use `logging.getLogger(__name__)`.
    - [ ] [Medium] Ensure users are guided on how to configure logging via library functions or `basicConfig` for their application. (Self-note: Partly done, this is for a final review).

## User Experience

- [ ] [Medium] Create example notebooks and tutorials
  - [ ] [Medium] Basic model building examples
  - [ ] [Medium] Advanced analysis examples
  - [ ] [Low] Real-world case studies

- [ ] [Medium] Update documentation
  - [ ] [Medium] User guide
  - [ ] [High] API reference
  - [ ] [Medium] Best practices guide 

## Notebooks
- [ ] [Medium] Review and update Jupyter Notebooks (.ipynb files) in the 'nb/' directory:
    - [ ] [Medium] Remove or replace `warnings.filterwarnings('ignore')`.
    - [ ] [Medium] Ensure notebooks run correctly with the latest code.
    - [ ] [Medium] Align notebook code with library's best practices (e.g., logging).
