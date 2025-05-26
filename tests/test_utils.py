"""
Tests for utility module functionality.
"""

import unittest
import pandas as pd
import os
from pycost import utils

class TestCostToolsAccessor(unittest.TestCase):
    """Test cases for the CostTools pandas accessor."""
    
    def setUp(self):
        """Set up test data."""
        # Create a test dataframe with fiscal year columns
        self.df = pd.DataFrame({
            'Project': ['A', 'B', 'C'],
            'Category': ['X', 'Y', 'Z'],
            'FY2020': [100, 200, 300],
            'FY2021': [110, 220, 330],
            'FY2022': [120, 240, 360],
            'start_date': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'end_value': [1.1, 2.2, 3.3],
            'middle_part': ['foo', 'bar', 'baz']
        })
    
    def test_contains(self):
        """Test contains method."""
        # Get columns containing 'FY'
        result = self.df.ct.contains('FY')
        self.assertEqual(list(result.columns), ['FY2020', 'FY2021', 'FY2022'])
        
        # Test case insensitive search
        result = self.df.ct.contains('fy', case=False)
        self.assertEqual(list(result.columns), ['FY2020', 'FY2021', 'FY2022'])
    
    def test_starts_with(self):
        """Test starts_with method."""
        # Get columns starting with 'FY'
        result = self.df.ct.starts_with('FY')
        self.assertEqual(list(result.columns), ['FY2020', 'FY2021', 'FY2022'])
        
        # Get columns starting with 'start'
        result = self.df.ct.starts_with('start')
        self.assertEqual(list(result.columns), ['start_date'])
    
    def test_ends_with(self):
        """Test ends_with method."""
        # Get columns ending with 'date'
        result = self.df.ct.ends_with('date')
        self.assertEqual(list(result.columns), ['start_date'])
        
        # Get columns ending with 'value'
        result = self.df.ct.ends_with('value')
        self.assertEqual(list(result.columns), ['end_value'])
    
    def test_select(self):
        """Test select method."""
        # Select specific columns with plus notation
        result = self.df.ct.select('Project + Category')
        self.assertEqual(list(result.columns), ['Project', 'Category'])
        
        # Test with function calls
        result = self.df.ct.select('starts_with("FY") + Project')
        self.assertEqual(list(result.columns), ['FY2020', 'FY2021', 'FY2022', 'Project'])
        
        # Test with negative selection
        result = self.df.ct.select('starts_with("FY") - FY2021')
        self.assertEqual(list(result.columns), ['FY2020', 'FY2022'])
        
    def test_get_fys(self):
        """Test get_fys method."""
        # Get boolean mask for FY columns
        result = self.df.ct.get_fys()
        self.assertEqual(list(result), [False, False, True, True, True, False, False, False])
    
    def test_stack_fys(self):
        """Test stack_fys method."""
        # Stack fiscal year columns
        result = self.df.ct.stack_fys()
        
        # Should have Project, Category, FY, value columns
        self.assertEqual(set(result.columns), set(['Project', 'Category', 'FY', 'start_date', 'end_value', 'middle_part', 'value']))
        
        # Should have 9 rows (3 original rows * 3 FY columns)
        self.assertEqual(len(result), 9)
        
        # FY column should contain the FY values
        self.assertEqual(set(result['FY']), set(['FY2020', 'FY2021', 'FY2022']))


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for standalone utility functions."""
    
    def test_get_fys(self):
        """Test get_fys standalone function."""
        df = pd.DataFrame({
            'Project': ['A', 'B'],
            'FY2020': [100, 200],
            'FY2021': [110, 220],
            'Fiscal Year 2022': [120, 240]
        })
        
        result = utils.get_fys(df)
        self.assertEqual(list(result), [False, True, True, True])
        
        # Test with custom pattern
        result = utils.get_fys(df, 'Fiscal Year')
        self.assertEqual(list(result), [False, False, False, True])
    
    def test_stack_fys(self):
        """Test stack_fys standalone function."""
        df = pd.DataFrame({
            'Project': ['A', 'B'],
            'FY2020': [100, 200],
            'FY2021': [110, 220]
        })
        
        result = utils.stack_fys(df)
        
        # Should have Project, FY, value columns
        self.assertEqual(set(result.columns), set(['Project', 'FY', 'value']))
        
        # Should have 4 rows (2 original rows * 2 FY columns)
        self.assertEqual(len(result), 4)
        
        # Check values
        self.assertEqual(result.loc[result['Project'] == 'A', 'value'].sum(), 210)
        self.assertEqual(result.loc[result['Project'] == 'B', 'value'].sum(), 420)
    
    def test_get_imports(self):
        """Test get_imports function."""
        # This is hard to test directly, so just check it returns something
        result = utils.get_imports()
        self.assertIsInstance(result, list)
        
        # Should have some entries because we've imported pandas at least
        self.assertGreater(len(result), 0)

if __name__ == '__main__':
    unittest.main() 