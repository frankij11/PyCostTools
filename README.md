# Py Cost Tools
A suite of tools to help cost estimators

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frankij11/PyCostTools/HEAD)

## Installation
* requires a copy of git 
* otherwise need to clone / copy to desktop then run setup.py
```
pip install git+https://github.com/frankij11/PyCostTools.git#egg=pycost
```

## Usage

```Python
import pycost as ct
ct.inflation.BYtoBY(Index = "APN", FromYR = 2020, ToYR = 2025, Cost = 1)
```
