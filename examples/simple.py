"""
Simple example for the PyCost framework.

This example demonstrates the basic usage of cost estimation models.
"""

import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import basic model types
from pycost.cost.core.base import GlobalInputs
from pycost.cost.core.inventory import Inventory
from pycost.cost.templates.development import Development
from pycost.cost.templates.production import Production

def main():
    """Run a simple cost estimation example."""
    # Create global inputs with custom settings
    global_inputs = GlobalInputs(
        long_name="Example Program",
        short_name="EX",
        base_year=2023,
        dol_units=1_000_000  # Display in millions
    )
    
    # Create a development model
    dev_model = Development(
        cost=1000,              # $1000M development cost
        duration=5,             # 5 years
        start_year=2024,        # Start in 2024
        global_inputs=global_inputs,
        meta={
            'Analyst': 'Example User',
            'Element': 'RDT&E'
        }
    )
    
    # Create an inventory profile
    inventory = Inventory(
        profile=pd.DataFrame({
            'FY': range(2026, 2036),
            'quantity': [5] * 5 + [10] * 5
        }),
        delivery_cycle=2      # 2 years from procurement to delivery
    )
    
    # Create a production model
    prod_model = Production(
        first_unit_cost=50,     # $50M for first unit
        learning_curve=0.85,    # 85% learning curve
        rate_curve=0.95,        # 95% rate curve
        inventory=inventory,
        global_inputs=global_inputs,
        meta={
            'Analyst': 'Example User',
            'Element': 'Production'
        }
    )
    
    # Calculate costs
    dev_model.calc_cost()
    prod_model.calc_cost()
    
    # Display results
    print("\n--- Development Costs ---")
    print(dev_model.cost_estimate[['FY', 'value_cp']].to_string(index=False))
    print(f"Total Development: ${dev_model.total_cost_cp:.1f}M")
    
    print("\n--- Production Costs ---")
    print(prod_model.cost_estimate[['FY', 'quantity', 'value_cp']].to_string(index=False))
    print(f"Total Production: ${prod_model.total_cost_cp:.1f}M")
    
    print(f"\nTotal Program: ${dev_model.total_cost_cp + prod_model.total_cost_cp:.1f}M")
    
if __name__ == "__main__":
    main()