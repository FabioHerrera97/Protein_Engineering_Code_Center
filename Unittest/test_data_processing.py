import unittest
import pandas as pd
import numpy as np
import os
from Src.data_processing import DataProcessing

class TestDataProcessing(unittest.TestCase):
    """
    This class contains unit tests for the data processing functions.
    """

    def setUp(self):
        """
        Set up the test data and variables.
        """
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        self.threshold = 0.1
        self.output_file = '../Reports/outlier_report.txt'

    def test_detect_outliers(self):
        DataProcessing.create_outlier_report(self.data, self.threshold, self.output_file)
        self.assertTrue(os.path.exists(self.output_file))
        os.remove(self.output_file) # Clean up the output file after the test is done

    def test_transform_to_logarithmic(self):
        transformed_data = DataProcessing.transform_to_logarithmic(self.data)
        expected_data = np.log(self.data)
        pd.testing.assert_frame_equal(transformed_data, expected_data)

if __name__ == '__main__':
    unittest.main()