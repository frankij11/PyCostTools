#!/usr/bin/env python3
"""
PyCost PyPI Deployment Script

This script helps deploy the PyCost package to PyPI following best practices.
It includes checks for package quality, builds the distribution, and uploads to PyPI.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    
    return result


def check_prerequisites():
    """Check that all prerequisites are installed."""
    print("Checking prerequisites...")
    
    required_packages = ["build", "twine", "pytest"]
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Install them with: pip install build twine pytest")
        sys.exit(1)
    
    print("✓ All prerequisites are installed")


def clean_build_artifacts():
    """Clean up build artifacts from previous builds."""
    print("Cleaning build artifacts...")
    
    artifacts = ["build", "dist", "*.egg-info"]
    for pattern in artifacts:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Removed directory: {path}")
            else:
                path.unlink()
                print(f"Removed file: {path}")
    
    print("✓ Build artifacts cleaned")


def run_tests():
    """Run the test suite."""
    print("Running tests...")
    
    # Run pytest
    result = run_command([sys.executable, "-m", "pytest", "tests/", "-v"], check=False)
    
    if result.returncode != 0:
        print("❌ Tests failed. Please fix issues before deploying.")
        return False
    
    print("✓ All tests passed")
    return True


def run_examples():
    """Run all examples to ensure they work."""
    print("Running examples...")
    
    result = run_command([sys.executable, "test_examples.py"], check=False)
    
    if result.returncode != 0:
        print("❌ Examples failed. Please fix issues before deploying.")
        return False
    
    print("✓ All examples passed")
    return True


def check_package_quality():
    """Run quality checks on the package."""
    print("Running package quality checks...")
    
    # Check if we can import the package
    try:
        import pycost
        print(f"✓ Package imports successfully (version {pycost.__version__})")
    except ImportError as e:
        print(f"❌ Cannot import package: {e}")
        return False
    
    # Check for required files
    required_files = ["README.md", "LICENSE.txt", "setup.py", "pyproject.toml"]
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Missing required file: {file}")
            return False
    
    print("✓ Package quality checks passed")
    return True


def build_package():
    """Build the package distribution."""
    print("Building package...")
    
    # Build using the build module
    run_command([sys.executable, "-m", "build"])
    
    # Check that dist files were created
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("❌ No dist directory created")
        return False
    
    wheel_files = list(dist_dir.glob("*.whl"))
    tar_files = list(dist_dir.glob("*.tar.gz"))
    
    if not wheel_files:
        print("❌ No wheel file created")
        return False
    
    if not tar_files:
        print("❌ No source distribution created")
        return False
    
    print(f"✓ Package built successfully:")
    for file in wheel_files + tar_files:
        print(f"  - {file}")
    
    return True


def check_package_with_twine():
    """Check the package with twine."""
    print("Checking package with twine...")
    
    result = run_command([sys.executable, "-m", "twine", "check", "dist/*"], check=False)
    
    if result.returncode != 0:
        print("❌ Twine check failed")
        return False
    
    print("✓ Twine check passed")
    return True


def upload_to_test_pypi():
    """Upload to Test PyPI."""
    print("Uploading to Test PyPI...")
    
    result = run_command([
        sys.executable, "-m", "twine", "upload",
        "--repository", "testpypi",
        "dist/*"
    ], check=False)
    
    if result.returncode != 0:
        print("❌ Upload to Test PyPI failed")
        return False
    
    print("✓ Successfully uploaded to Test PyPI")
    return True


def upload_to_pypi():
    """Upload to PyPI."""
    print("Uploading to PyPI...")
    
    result = run_command([
        sys.executable, "-m", "twine", "upload",
        "dist/*"
    ], check=False)
    
    if result.returncode != 0:
        print("❌ Upload to PyPI failed")
        return False
    
    print("✓ Successfully uploaded to PyPI")
    return True


def main():
    """Main deployment function."""
    print("=" * 60)
    print("PyCost PyPI Deployment Script")
    print("=" * 60)
    
    # Parse command line arguments
    test_only = "--test" in sys.argv
    skip_tests = "--skip-tests" in sys.argv
    
    if test_only:
        print("Running in TEST mode - will upload to Test PyPI only")
    
    # Step 1: Check prerequisites
    check_prerequisites()
    
    # Step 2: Clean build artifacts
    clean_build_artifacts()
    
    # Step 3: Run tests (unless skipped)
    if not skip_tests:
        if not run_tests():
            sys.exit(1)
        
        if not run_examples():
            sys.exit(1)
    else:
        print("⚠ Skipping tests as requested")
    
    # Step 4: Check package quality
    if not check_package_quality():
        sys.exit(1)
    
    # Step 5: Build package
    if not build_package():
        sys.exit(1)
    
    # Step 6: Check with twine
    if not check_package_with_twine():
        sys.exit(1)
    
    # Step 7: Upload
    if test_only:
        if not upload_to_test_pypi():
            sys.exit(1)
        print("\n🎉 Package successfully deployed to Test PyPI!")
        print("You can install it with:")
        print("pip install --index-url https://test.pypi.org/simple/ pycost")
    else:
        # Ask for confirmation before uploading to real PyPI
        response = input("\nReady to upload to PyPI. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Deployment cancelled.")
            sys.exit(0)
        
        if not upload_to_pypi():
            sys.exit(1)
        
        print("\n🎉 Package successfully deployed to PyPI!")
        print("You can install it with:")
        print("pip install pycost")
    
    print("\nDeployment completed successfully!")


if __name__ == "__main__":
    main() 