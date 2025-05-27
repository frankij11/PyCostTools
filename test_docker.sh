#!/bin/bash
# PyCost Docker Testing Script for Unix-like systems
# Tests the package across multiple Python versions using Docker

set -e  # Exit on any error

echo "========================================"
echo "PyCost Multi-Version Docker Testing"
echo "========================================"

PYTHON_VERSIONS="3.8 3.9 3.10 3.11 3.12"
IMAGE_NAME="pycost-test"
FAILED_VERSIONS=""

echo "Starting tests for Python versions: $PYTHON_VERSIONS"
echo

for version in $PYTHON_VERSIONS; do
    echo "----------------------------------------"
    echo "Testing Python $version"
    echo "----------------------------------------"
    
    # Build the Docker image for this Python version
    echo "Building Docker image for Python $version..."
    if ! docker build --build-arg PYTHON_VERSION=$version -t $IMAGE_NAME:$version . --target base; then
        echo "ERROR: Failed to build Docker image for Python $version"
        FAILED_VERSIONS="$FAILED_VERSIONS $version"
        continue
    fi
    
    # Run unit tests
    echo "Running unit tests for Python $version..."
    if ! docker run --rm $IMAGE_NAME:$version python -m pytest tests/ -v --tb=short; then
        echo "ERROR: Unit tests failed for Python $version"
        FAILED_VERSIONS="$FAILED_VERSIONS $version"
        continue
    fi
    
    # Test examples
    echo "Testing examples for Python $version..."
    if ! docker build --build-arg PYTHON_VERSION=$version -t $IMAGE_NAME:$version-examples . --target test-examples; then
        echo "ERROR: Examples failed for Python $version"
        FAILED_VERSIONS="$FAILED_VERSIONS $version"
        continue
    fi
    
    # Test package installation
    echo "Testing package installation for Python $version..."
    if ! docker run --rm $IMAGE_NAME:$version python -c "import pycost; print(f'PyCost {pycost.__version__} imported successfully on Python $version')"; then
        echo "ERROR: Package import failed for Python $version"
        FAILED_VERSIONS="$FAILED_VERSIONS $version"
        continue
    fi
    
    echo "SUCCESS: All tests passed for Python $version"
    echo
done

echo "========================================"
echo "Test Summary"
echo "========================================"

if [ -z "$FAILED_VERSIONS" ]; then
    echo "✓ All Python versions passed all tests!"
    echo "The package is ready for PyPI deployment."
    exit 0
else
    echo "✗ The following Python versions failed:$FAILED_VERSIONS"
    echo "Please review the errors above before deploying."
    exit 1
fi

echo
echo "To clean up Docker images, run:"
echo "docker rmi $IMAGE_NAME:3.8 $IMAGE_NAME:3.9 $IMAGE_NAME:3.10 $IMAGE_NAME:3.11 $IMAGE_NAME:3.12"
echo 