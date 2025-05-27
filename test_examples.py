#!/usr/bin/env python3
"""
Test runner for PyCost examples.

This script runs all examples in the examples/ directory to ensure they work
properly across different Python versions and environments.
"""

import os
import sys
import subprocess
import traceback
from pathlib import Path
from typing import List, Tuple


def run_example(example_path: Path) -> Tuple[bool, str]:
    """
    Run a single example and return success status and output.
    
    Args:
        example_path: Path to the example file
        
    Returns:
        Tuple of (success, output_message)
    """
    try:
        print(f"Running {example_path.name}...")
        
        # Change to examples directory to run the example
        original_cwd = os.getcwd()
        os.chdir(example_path.parent)
        
        # Run the example
        result = subprocess.run(
            [sys.executable, example_path.name],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        # Restore original directory
        os.chdir(original_cwd)
        
        if result.returncode == 0:
            return True, f"✓ {example_path.name} completed successfully"
        else:
            error_msg = f"✗ {example_path.name} failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f"\nSTDERR: {result.stderr}"
            if result.stdout:
                error_msg += f"\nSTDOUT: {result.stdout}"
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        os.chdir(original_cwd)
        return False, f"✗ {example_path.name} timed out after 60 seconds"
    except Exception as e:
        os.chdir(original_cwd)
        return False, f"✗ {example_path.name} failed with exception: {str(e)}"


def find_examples() -> List[Path]:
    """Find all Python example files."""
    examples_dir = Path(__file__).parent / "examples"
    if not examples_dir.exists():
        print(f"Examples directory not found: {examples_dir}")
        return []
    
    # Find all .py files in examples directory
    examples = list(examples_dir.glob("*.py"))
    
    # Filter out any files that shouldn't be run
    excluded = {"__init__.py", "test_examples.py"}
    examples = [ex for ex in examples if ex.name not in excluded]
    
    return sorted(examples)


def test_package_import():
    """Test that the package can be imported properly."""
    try:
        import pycost
        print(f"✓ PyCost {pycost.__version__} imported successfully")
        
        # Test importing main modules
        from pycost import utils, inflation, learn
        print("✓ Core modules imported successfully")
        
        # Test importing analysis modules
        try:
            from pycost import analysis
            print("✓ Analysis modules imported successfully")
        except ImportError as e:
            print(f"⚠ Warning: Could not import analysis modules: {e}")
        
        # Test importing cost modules
        try:
            from pycost import cost
            print("✓ Cost modules imported successfully")
        except ImportError as e:
            print(f"⚠ Warning: Could not import cost modules: {e}")
            
        return True
        
    except Exception as e:
        print(f"✗ Failed to import pycost: {e}")
        traceback.print_exc()
        return False


def main():
    """Main test runner."""
    print("=" * 50)
    print("PyCost Examples Test Runner")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Test package import first
    print("Testing package import...")
    if not test_package_import():
        print("Package import failed. Cannot proceed with examples.")
        sys.exit(1)
    print()
    
    # Find and run examples
    examples = find_examples()
    if not examples:
        print("No examples found to test.")
        sys.exit(1)
    
    print(f"Found {len(examples)} examples to test:")
    for example in examples:
        print(f"  - {example.name}")
    print()
    
    # Run each example
    results = []
    for example in examples:
        success, message = run_example(example)
        results.append((example.name, success, message))
        print(message)
    
    print()
    print("=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"Successful: {successful}/{total}")
    
    if successful == total:
        print("🎉 All examples passed!")
        sys.exit(0)
    else:
        print("❌ Some examples failed:")
        for name, success, message in results:
            if not success:
                print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main() 