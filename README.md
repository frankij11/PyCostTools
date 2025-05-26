# PyCost - Python Tools for Cost Estimation

PyCost is a Python package that provides tools and utilities for cost estimation, learning curve calculations, inflation adjustments, and other cost analysis tasks commonly used in program management.

## Features

- **Inflation Adjustments**: Convert costs between base years and then-years using standard indices
- **Learning Curve Calculations**: Implement learning and rate curve effects in cost estimates
- **DataFrame Extensions**: Pandas extensions for manipulating cost data tables
- **Cost Models**: Framework for building cost models with reactive calculations
- **Interactive Visualization**: Integration with Panel for interactive dashboards

## Installation

```bash
pip install pycost
```

## Usage Examples

### Inflation Adjustments

```python
from pycost import inflation

# Convert $100 in 2018 dollars to 2020 dollars using RDT&E index
inflation.BYtoBY('RDT&E', '2018', '2020', 100)

# Convert base year to then year
inflation.BYtoTY('RDT&E', '2018', '2020', 100)
```

### Learning Curve Calculations

```python
from pycost import learn

# Calculate learning curve effect
# T1 = $100, 90% learning curve, 95% rate curve, 10 units at rate of 2/month
learn.learn_curve(100, 0.9, 0.95, 10, 2)

# Calculate midpoint for units 1-10 with 90% learning curve
learn.asher_midpoint(1, 10, 0.9)
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

# Stack fiscal years into long format
stacked = df.ct.stack_fys()
```

## Development

To set up the development environment:

```bash
git clone https://github.com/frankij11/PyCostTools.git
cd PyCostTools
pip install -e .
pip install -r requirements-dev.txt
```

### Running Tests

```bash
python -m unittest discover tests
```

## License

MIT License - see LICENSE.txt for details.

## Acknowledgments

This project was developed to provide a Python alternative to traditional Excel-based cost estimation tools.
