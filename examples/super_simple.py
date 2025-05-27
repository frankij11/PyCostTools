"""
Super simple example for the PyCost framework.

This example demonstrates the most basic usage of PyCost utilities.
"""

import pandas as pd
import numpy as np

# Import basic utilities
from pycost import utils, learn

def main():
    """Run a super simple example."""
    print("PyCost Super Simple Example")
    print("=" * 40)
    
    # Example 1: Learning curve calculations
    print("\n1. Learning Curve Calculations:")
    first_unit_cost = 100
    learning_curve = 0.9
    rate_curve = 0.95
    quantity = 10
    rate = 2
    
    total_cost = learn.learn_curve(first_unit_cost, learning_curve, rate_curve, quantity, rate)
    print(f"   Total cost for {quantity} units: ${total_cost:.2f}")
    
    # Calculate midpoint using lc_midpoint (which works)
    midpoint = learn.lc_midpoint(1, 10, 0.9)
    print(f"   Midpoint unit for 1-10 with 90% learning: {midpoint:.2f}")
    
    # Example 2: DataFrame utilities
    print("\n2. DataFrame Utilities:")
    df = pd.DataFrame({
        'Project': ['A', 'B', 'C'],
        'FY2020': [100, 200, 150],
        'FY2021': [110, 220, 165],
        'FY2022': [120, 240, 180]
    })
    
    print("   Original DataFrame:")
    print(df.to_string(index=False))
    
    # Use PyCost DataFrame extensions
    fy_columns = df.ct.contains('FY')
    print(f"\n   Fiscal year columns: {fy_columns}")
    
    # Stack fiscal years
    stacked = df.ct.stack_fys()
    print("\n   Stacked format:")
    print(stacked.head().to_string(index=False))
    
    # Example 3: More learning curve calculations
    print("\n3. More Learning Curve Examples:")
    
    # Different learning curves
    learning_curves = [0.8, 0.85, 0.9, 0.95]
    for lc in learning_curves:
        cost = learn.learn_curve(100, lc, 1.0, 10, 1)
        print(f"   {int(lc*100)}% learning curve: ${cost:.2f}")
    
    print("\nSUCCESS: Super simple example completed successfully!")

if __name__ == "__main__":
    main()


    
    #print(labor.calc_cost())
    #print(labor.calc_cost_uncertainty())
    #print(labor.calc_cost_metadata())
    
    

