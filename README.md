# PyCost - Python Tools for Cost Estimation

[![PyPI version](https://badge.fury.io/py/pycost.svg)](https://badge.fury.io/py/pycost)
[![Python versions](https://img.shields.io/pypi/pyversions/pycost.svg)](https://pypi.org/project/pycost/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyCost is a comprehensive Python package that provides tools and utilities for cost estimation, learning curve calculations, inflation adjustments, and other cost analysis tasks commonly used in program management and financial analysis.

## 🚀 Features

- **💰 Inflation Adjustments**: Convert costs between base years and then-years using standard indices
- **📈 Learning Curve Calculations**: Implement learning and rate curve effects in cost estimates
- **📊 DataFrame Extensions**: Pandas extensions for manipulating cost data tables
- **🏗️ Cost Models**: Framework for building cost models with reactive calculations
- **📱 Interactive Visualization**: Integration with Panel for interactive dashboards
- **🔧 Utilities**: Comprehensive set of utilities for cost analysis workflows

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install pycost
```

### From Source

```bash
git clone https://github.com/frankij11/PyCostTools.git
cd PyCostTools
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/frankij11/PyCostTools.git
cd PyCostTools
pip install -e ".[dev]"
```

## 🏃‍♂️ Quick Start

### Basic Usage

```python
import pycost

# Check version
print(f"PyCost version: {pycost.__version__}")

# Import main modules
from pycost import inflation, learn, utils
```

### Inflation Adjustments

```python
from pycost import inflation

# Convert $100 in 2018 dollars to 2020 dollars using RDT&E index
future_cost = inflation.BYtoBY('RDT&E', '2018', '2020', 100)
print(f"$100 in 2018 = ${future_cost:.2f} in 2020")

# Convert base year to then year
then_year_cost = inflation.BYtoTY('RDT&E', '2018', '2020', 100)
print(f"Base year cost: ${then_year_cost:.2f}")
```

### Learning Curve Calculations

```python
from pycost import learn

# Calculate learning curve effect
# T1 = $100, 90% learning curve, 95% rate curve, 10 units at rate of 2/month
total_cost = learn.learn_curve(100, 0.9, 0.95, 10, 2)
print(f"Total cost with learning: ${total_cost:.2f}")

# Calculate midpoint for units 1-10 with 90% learning curve
midpoint = learn.asher_midpoint(1, 10, 0.9)
print(f"Midpoint unit: {midpoint:.2f}")
```

### DataFrame Extensions

```python
import pandas as pd
from pycost import utils

# Create a DataFrame with fiscal year columns
df = pd.DataFrame({
    'Project': ['A', 'B'],
    'FY2020': [100, 200],
    'FY2021': [110, 220],
    'FY2022': [120, 240]
})

# Select fiscal year columns
fy_cols = df.ct.contains('FY')
print("Fiscal year columns:", fy_cols)

# Stack fiscal years into long format
stacked = df.ct.stack_fys()
print(stacked.head())
```

## 📚 Documentation

### Core Modules

- **`pycost.inflation`**: Inflation adjustment utilities
- **`pycost.learn`**: Learning curve calculations
- **`pycost.utils`**: DataFrame extensions and utilities
- **`pycost.cost`**: Cost modeling framework
- **`pycost.analysis`**: Advanced analysis tools

### Examples

The `examples/` directory contains comprehensive examples:

- `super_simple.py`: Basic package usage
- `simple.py`: Cost estimation models
- `demo_program.py`: Program-level cost analysis
- `manufacturing_lot_example.py`: Manufacturing cost modeling
- `model_analysis_example.py`: Advanced model analysis

Run examples:

```bash
cd examples
python simple.py
```

Or test all examples:

```bash
python test_examples.py
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pycost --cov-report=html

# Run specific test file
pytest tests/test_inflation.py -v
```

### Test Examples

```bash
# Test all examples work correctly
python test_examples.py
```

### Docker Testing

Test across multiple Python versions using Docker:

```bash
# Windows
test_docker.bat

# Unix/Linux/macOS
./test_docker.sh
```

## 🚀 Development

### Setting Up Development Environment

```bash
git clone https://github.com/frankij11/PyCostTools.git
cd PyCostTools

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Code Quality

```bash
# Format code
black pycost/ tests/ examples/

# Lint code
flake8 pycost/ tests/ examples/

# Type checking
mypy pycost/
```

### Building and Deployment

```bash
# Test deployment to Test PyPI
python deploy.py --test

# Deploy to PyPI (requires credentials)
python deploy.py
```

## 🐳 Docker

### Build Docker Image

```bash
docker build -t pycost .
```

### Run Tests in Docker

```bash
# Run unit tests
docker run --rm pycost

# Test examples
docker build -t pycost-examples . --target test-examples
```

## 📋 Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib
- seaborn
- panel >= 0.14.0
- param >= 1.12.0
- holoviews >= 1.15.0
- hvplot >= 0.7.0
- networkx >= 2.6.0
- patsy

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## 🙏 Acknowledgments

This project was developed to provide a Python alternative to traditional Excel-based cost estimation tools, making cost analysis more reproducible, scalable, and maintainable.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/frankij11/PyCostTools/issues)
- **Documentation**: [GitHub Repository](https://github.com/frankij11/PyCostTools)
- **Email**: kevinfjoy@gmail.com

## 🔄 Changelog

See [CHANGES.md](CHANGES.md) for a detailed changelog.

---

**Made with ❤️ for the cost estimation community**
