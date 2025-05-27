# Contributing to PyCost

Thank you for considering contributing to PyCost! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

1. **Fork the repository** on GitHub
2. **Clone your fork** to your local machine
3. **Create a branch** for your changes
4. **Make your changes** and commit them with clear commit messages
5. **Push your changes** to your fork
6. **Submit a pull request** to the main repository

## Development Setup

To set up your development environment:

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/PyCostTools.git
cd PyCostTools

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

## Testing

Before submitting a pull request, please run the tests:

```bash
python -m unittest discover tests
```

Make sure to add tests for any new functionality you add.

## Code Style

Please follow these guidelines for code style:

- Use PEP 8 for code formatting
- Add docstrings for all functions, classes, and methods
- Use type hints where appropriate
- Keep lines under 100 characters
- Consider using `black` for code formatting, `flake8` for linting, and `mypy` for type checking to maintain code quality and consistency. Configuration files for these tools may be added to the project.

## Submitting Changes

When submitting a pull request:

1. Describe what your changes do and why they should be included
2. Include any relevant issue numbers in the PR description
3. Make sure all tests pass
4. Make sure your code lints without errors

## Documentation

- Update the README.md file if necessary
- Add or update docstrings for all functions, classes, and methods
- Consider whether your change requires updates to any examples

## Questions?

If you have any questions, please open an issue on GitHub. 