"""
Tests for inflation module functionality.
"""

import unittest
import numpy as np
import pandas as pd
from pycost import inflation

class TestInflation(unittest.TestCase):
    """Test cases for inflation conversion functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create a small test inflation table
        data = {
            'Indice': ['RDT&E', 'RDT&E', 'RDT&E', 'PROC', 'PROC', 'PROC'],
            'Year': ['2018', '2019', '2020', '2018', '2019', '2020'],
            'Raw': [0.9, 0.95, 1.0, 0.92, 0.96, 1.0],
            'Weighted': [0.88, 0.94, 1.0, 0.91, 0.95, 1.0]
        }
        self.test_inflation_table = pd.DataFrame(data)
    
    def test_BYtoBY(self):
        """Test base year to base year conversion."""
        # Convert 100 in 2018 dollars to 2020 dollars using RDT&E index
        result = inflation.BYtoBY('RDT&E', '2018', '2020', 100, self.test_inflation_table)
        # 100 / 0.9 * 1.0 = 111.11...
        self.assertAlmostEqual(result[0], 111.11, places=2)
    
    def test_BYtoTY(self):
        """Test base year to then year conversion."""
        # Convert 100 in 2018 dollars to 2020 then-year dollars using RDT&E index
        result = inflation.BYtoTY('RDT&E', '2018', '2020', 100, self.test_inflation_table)
        # 100 / 0.9 * 1.0 = 111.11...
        self.assertAlmostEqual(result[0], 111.11, places=2)
    
    def test_TYtoBY(self):
        """Test then year to base year conversion."""
        # Convert 100 in 2018 then-year dollars to 2020 base-year dollars using RDT&E index
        result = inflation.TYtoBY('RDT&E', '2018', '2020', 100, self.test_inflation_table)
        # 100 / 0.88 * 1.0 = 113.64...
        self.assertAlmostEqual(result[0], 113.64, places=2)
    
    def test_TYtoTY(self):
        """Test then year to then year conversion."""
        # Convert 100 in 2018 then-year dollars to 2020 then-year dollars using RDT&E index
        result = inflation.TYtoTY('RDT&E', '2018', '2020', 100, self.test_inflation_table)
        # 100 / 0.88 * 1.0 = 113.64...
        self.assertAlmostEqual(result[0], 113.64, places=2)
    
    def test_different_indices(self):
        """Test conversion between different indices."""
        # Convert 100 in 2018 PROC dollars to 2020 RDT&E dollars
        result = inflation.BYtoBY('PROC', '2018', '2020', 100, self.test_inflation_table)
        # 100 / 0.92 * 1.0 = 108.70...
        self.assertAlmostEqual(result[0], 108.70, places=2)

if __name__ == '__main__':
    unittest.main() 