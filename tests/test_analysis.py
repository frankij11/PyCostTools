"""
Tests for analysis module functionality.
"""

import unittest
import pandas as pd
import numpy as np
from pycost import jic

class TestAnalysis(unittest.TestCase):
    """Test cases for analysis module functions."""
    
    def setUp(self):
        """Set up test data."""
        self.df = jic
        self.df['Year'] = pd.to_numeric(self.df['Year'], errors='coerce')
    
    def test_jic_loaded(self):
        """Test that JIC data is loaded correctly."""
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertIn('Raw', self.df.columns)
        self.assertIn('Weighted', self.df.columns)
        self.assertIn('Year', self.df.columns)
    
    def test_jic_structure(self):
        """Test the structure of JIC data."""
        # Check that key columns exist
        required_columns = ['Version', 'Service', 'Indice', 'Year', 'Raw', 'Weighted']
        for col in required_columns:
            self.assertIn(col, self.df.columns)
        
        # Check that there are multiple years
        years = self.df['Year'].unique()
        self.assertGreater(len(years), 5)
        
        # Check that there are multiple indices
        indices = self.df['Indice'].unique()
        self.assertGreater(len(indices), 3)

if __name__ == '__main__':
    unittest.main()
