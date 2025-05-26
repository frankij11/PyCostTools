"""
Tests for learning curve module functionality.
"""

import unittest
import numpy as np
import pandas as pd
from pycost import learn

class TestLearningCurve(unittest.TestCase):
    """Test cases for learning curve functions."""
    
    def test_asher_midpoint(self):
        """Test Asher midpoint calculation."""
        # For 0.9 learning curve slope and units 1 to 10
        result = learn.asher_midpoint(1, 10, 0.9)
        # Actual calculation value (not a hard-coded expectation)
        expected = 1.0/(np.log(0.9)/np.log(2) + 1.0) * (10**(np.log(0.9)/np.log(2) + 1.0) - 1) / (10**(np.log(0.9)/np.log(2)) - 1)
        self.assertAlmostEqual(result, expected, places=6)
        
    def test_asher_midpoint_error(self):
        """Test error handling in Asher midpoint."""
        # Slope must be between 0 and 1
        with self.assertRaises(ValueError):
            learn.asher_midpoint(1, 10, 1.1)
        
        with self.assertRaises(ValueError):
            learn.asher_midpoint(1, 10, 0)
    
    def test_lc_midpoint(self):
        """Test learning curve midpoint calculation."""
        # Simple midpoint with no learning effect
        result = learn.lc_midpoint(100, 80, 1)
        expected = (100 + 80 + 2 * np.sqrt(100 * 80)) / 4
        self.assertAlmostEqual(result, expected)
    
    def test_learn_curve(self):
        """Test learning curve calculation."""
        # First unit cost = 100, learning curve = 90%, rate curve = 95%
        # 10 units at a rate of 2 per month
        result = learn.learn_curve(100, 0.9, 0.95, 10, 2)
        
        # Calculate expected value directly
        expected = 100 * (10**(np.log(0.9)/np.log(2))) * (2**(np.log(0.95)/np.log(2)))
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_lc_alias(self):
        """Test that lc is an alias for learn_curve."""
        # Same parameters as test_learn_curve
        result1 = learn.learn_curve(100, 0.9, 0.95, 10, 2)
        result2 = learn.lc(100, 0.9, 0.95, 10, 2)
        
        # Results should be identical
        self.assertEqual(result1, result2)
    
    def test_lc_prep(self):
        """Test learning curve data preparation."""
        # Create test dataframe
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'B'],
            'subgroup': [1, 2, 1, 2, 3],
            'value': [10, 15, 8, 12, 5]
        })
        
        result = learn.lc_prep(df, ['group', 'subgroup'], 'value')
        
        # Check that result has all the required columns
        self.assertTrue(all(col in result.columns for col in ['First', 'Last', 'midpoint', 'share_qty']))
        
        # Check that group A has two rows with correct quantities
        group_a = result[result['group'] == 'A']
        self.assertEqual(len(group_a), 2)
        self.assertEqual(group_a['share_qty'].sum(), 25)
        
        # Check First/Last calculation for the first group
        self.assertEqual(group_a.iloc[0]['First'], 1)
        self.assertEqual(group_a.iloc[0]['Last'], 10)
        self.assertEqual(group_a.iloc[1]['First'], 11)
        self.assertEqual(group_a.iloc[1]['Last'], 25)

if __name__ == '__main__':
    unittest.main() 