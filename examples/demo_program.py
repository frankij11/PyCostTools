"""
Demo program for the PyCost framework.

This example demonstrates how to create and run a basic cost estimation model
combining development and production phases.
"""

import logging
from typing import Optional
import pandas as pd

# from pycost.cost.utils.logging import setup_logging
# setup_logging(log_level=logging.INFO) # Alternative to basicConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import necessary classes from pycost
from pycost.cost.core.parent_models import ParentModel
from pycost.cost.core.inventory import Inventory
from pycost.cost.templates.development import Development
from pycost.cost.templates.production import Production
from pycost.cost.utils.ui import ModelApp


class DemoProgram(ParentModel):
    """
    Example program combining development and production models.
    
    This class demonstrates how to combine multiple cost models
    into a complete program estimate with both development and
    production phases.
    
    Attributes:
        inventory (Inventory): Shared inventory object used across models
    """
    
    def __init__(self, inventory: Optional[Inventory] = None):
        """
        Initialize the demo program with development and production models.
        
        Args:
            inventory: Inventory profile for production.
                If None, a default Inventory instance will be created.
        """
        # Set up the inventory object
        self.inventory = inventory if inventory is not None else Inventory()
        
        # Create the component models
        self.models = [
            Development(
                cost=500, 
                duration=10, 
                start_year=2020,
                meta={
                    'Analyst': 'Demo User',
                    'Element': 'Development',
                }
            ),
            Production(
                inventory=self.inventory,
                meta={
                    'Analyst': 'Demo User',
                    'Element': 'Production',
                }
            )
        ]
        
        # Initialize the parent model
        super().__init__()


def main():
    """
    Main entry point for running the demo program.
    
    This function creates and displays a basic demo cost estimation model
    using the Panel interface.
    """
    # Create the demo program
    demo = DemoProgram()
    
    # Calculate costs
    demo.calc_cost()
    
    # Print a summary of the costs
    total_cp = demo.total_cost_cp
    logger.info(f"Total cost (constant price): {total_cp:,.2f}")
    
    # Create and display the UI
    app = ModelApp(model=demo)
    return app.show()
    
    
if __name__ == "__main__":
    main() 