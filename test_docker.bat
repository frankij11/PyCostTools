@echo off
REM PyCost Docker Testing Script for Windows
REM Tests the package across multiple Python versions using Docker

echo ========================================
echo PyCost Multi-Version Docker Testing
echo ========================================

set PYTHON_VERSIONS=3.8 3.9 3.10 3.11 3.12
set IMAGE_NAME=pycost-test
set FAILED_VERSIONS=

echo Starting tests for Python versions: %PYTHON_VERSIONS%
echo.

for %%v in (%PYTHON_VERSIONS%) do (
    echo ----------------------------------------
    echo Testing Python %%v
    echo ----------------------------------------
    
    REM Build the Docker image for this Python version
    echo Building Docker image for Python %%v...
    docker build --build-arg PYTHON_VERSION=%%v -t %IMAGE_NAME%:%%v . --target base
    
    if errorlevel 1 (
        echo ERROR: Failed to build Docker image for Python %%v
        set FAILED_VERSIONS=%FAILED_VERSIONS% %%v
        goto :continue
    )
    
    REM Run unit tests
    echo Running unit tests for Python %%v...
    docker run --rm %IMAGE_NAME%:%%v python -m pytest tests/ -v --tb=short
    
    if errorlevel 1 (
        echo ERROR: Unit tests failed for Python %%v
        set FAILED_VERSIONS=%FAILED_VERSIONS% %%v
        goto :continue
    )
    
    REM Test examples
    echo Testing examples for Python %%v...
    docker build --build-arg PYTHON_VERSION=%%v -t %IMAGE_NAME%:%%v-examples . --target test-examples
    
    if errorlevel 1 (
        echo ERROR: Examples failed for Python %%v
        set FAILED_VERSIONS=%FAILED_VERSIONS% %%v
        goto :continue
    )
    
    REM Test package installation
    echo Testing package installation for Python %%v...
    docker run --rm %IMAGE_NAME%:%%v python -c "import pycost; print(f'PyCost {pycost.__version__} imported successfully on Python %%v')"
    
    if errorlevel 1 (
        echo ERROR: Package import failed for Python %%v
        set FAILED_VERSIONS=%FAILED_VERSIONS% %%v
        goto :continue
    )
    
    echo SUCCESS: All tests passed for Python %%v
    echo.
    
    :continue
)

echo ========================================
echo Test Summary
echo ========================================

if "%FAILED_VERSIONS%"=="" (
    echo ✓ All Python versions passed all tests!
    echo The package is ready for PyPI deployment.
) else (
    echo ✗ The following Python versions failed: %FAILED_VERSIONS%
    echo Please review the errors above before deploying.
    exit /b 1
)

echo.
echo To clean up Docker images, run:
echo docker rmi %IMAGE_NAME%:3.8 %IMAGE_NAME%:3.9 %IMAGE_NAME%:3.10 %IMAGE_NAME%:3.11 %IMAGE_NAME%:3.12
echo.

pause 