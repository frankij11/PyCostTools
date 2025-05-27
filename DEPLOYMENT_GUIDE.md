# PyCost PyPI Deployment Guide

This guide provides step-by-step instructions for deploying the PyCost package to PyPI.

## 📋 Prerequisites

Before deploying, ensure you have:

1. **Python 3.8+** installed
2. **Git** for version control
3. **Docker** (optional, for multi-version testing)
4. **PyPI account** with API token
5. **Test PyPI account** (recommended for testing)

## 🔧 Setup

### 1. Install Build Dependencies

```bash
pip install build twine pytest
```

### 2. Set Up PyPI Credentials

Create a `.pypirc` file in your home directory:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-api-token-here
```

## 🧪 Testing Before Deployment

### 1. Run Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pycost --cov-report=html
```

### 2. Test Examples

```bash
# Test all examples
python test_examples.py
```

### 3. Docker Testing (Multi-Version)

```bash
# Windows
test_docker.bat

# Unix/Linux/macOS
./test_docker.sh
```

### 4. Manual Testing

```bash
# Test package import
python -c "import pycost; print(pycost.__version__)"

# Test basic functionality
cd examples
python super_simple.py
```

## 📦 Building the Package

### 1. Clean Previous Builds

```bash
# Remove old build artifacts
rm -rf build/ dist/ *.egg-info/
```

### 2. Build Distribution

```bash
# Build source and wheel distributions
python -m build
```

### 3. Check Package

```bash
# Verify package integrity
twine check dist/*
```

## 🚀 Deployment

### Option 1: Using the Deployment Script (Recommended)

```bash
# Test deployment to Test PyPI
python deploy.py --test

# Deploy to PyPI (after testing)
python deploy.py
```

### Option 2: Manual Deployment

#### Deploy to Test PyPI First

```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ pycost
```

#### Deploy to PyPI

```bash
# Upload to PyPI
twine upload dist/*

# Test installation from PyPI
pip install pycost
```

## 📋 Pre-Deployment Checklist

- [ ] All tests pass (`pytest tests/`)
- [ ] Examples work (`python test_examples.py`)
- [ ] Version number updated in `pycost/__init__.py`
- [ ] CHANGES.md updated with release notes
- [ ] README.md is current and accurate
- [ ] Package builds successfully (`python -m build`)
- [ ] Package passes twine check (`twine check dist/*`)
- [ ] Tested on Test PyPI
- [ ] Git repository is clean and committed

## 🔄 Version Management

### Update Version

1. Edit `pycost/__init__.py`:
   ```python
   __version__ = "0.2.0"  # Update version
   ```

2. Update `CHANGES.md` with release notes

3. Commit changes:
   ```bash
   git add .
   git commit -m "Release v0.2.0"
   git tag v0.2.0
   git push origin main --tags
   ```

## 🐳 Docker Testing

### Build and Test

```bash
# Build for Python 3.11
docker build -t pycost:3.11 --build-arg PYTHON_VERSION=3.11 .

# Run tests
docker run --rm pycost:3.11

# Test examples
docker build -t pycost:3.11-examples --build-arg PYTHON_VERSION=3.11 . --target test-examples
```

### Multi-Version Testing

The provided scripts test across Python 3.8-3.12:

- `test_docker.bat` (Windows)
- `test_docker.sh` (Unix/Linux/macOS)

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are in `requirements.txt`
2. **Build Failures**: Check `pyproject.toml` and `setup.py` configuration
3. **Upload Failures**: Verify PyPI credentials and package name availability
4. **Test Failures**: Fix failing tests before deployment

### Package Name Conflicts

If `pycost` is taken, consider alternatives:
- `pycost-estimation`
- `pycost-analysis`
- `pycost-framework`

Update the name in:
- `setup.py`
- `pyproject.toml`
- `README.md`

## 📚 Additional Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [setuptools Documentation](https://setuptools.pypa.io/)

## 🎯 Best Practices

1. **Always test on Test PyPI first**
2. **Use semantic versioning** (MAJOR.MINOR.PATCH)
3. **Keep detailed release notes**
4. **Test across multiple Python versions**
5. **Maintain backward compatibility when possible**
6. **Use GitHub releases** for version management
7. **Monitor package downloads and issues**

## 📞 Support

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/frankij11/PyCostTools/issues)
2. Review the deployment logs
3. Verify all prerequisites are met
4. Contact the maintainer: kevinfjoy@gmail.com 