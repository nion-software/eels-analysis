# run this from the command line using:
# cd EELSAnalysis
# python -m unittest test/Library_test.py

import numpy
import os
import sys
import unittest

# niondata must be available as a module.
# it can be added using something similar to
#   conda dev /path/to/niondata
from nion.data import Core
from nion.data import DataAndMetadata

sys.path.append(os.path.dirname(os.path.realpath(os.path.join(__file__, "..", ".."))))

# Note: EELSAnalysis is only available in sys.path above is appended with its _parent_ directory.
from EELSAnalysis import Library

class TestLibrary(unittest.TestCase):

    def setUp(self):
        """Common code for all tests can go here."""
        pass

    def tearDown(self):
        """Common code for all tests can go here."""
        pass

    def test_abc(self):
        depth, height, width = 100, 10, 10
        data = numpy.zeros((depth, height, width))
        for row in range(height):
            for column in range(width):
                data[:, row, column] = row + column + numpy.linspace(0, row + column, depth) + numpy.random.uniform(-1, 1, depth) / 2
        data_and_metadata = DataAndMetadata.DataAndMetadata.from_data(data)
        background_subtracted = Library.stacked_subtract_linear_background(data_and_metadata, (0.0, 1.0), (0.0, 1.0))
        self.assertEqual(background_subtracted.data_shape, (depth, height, width))
        self.assertTrue(numpy.all(background_subtracted.data < 1))
        self.assertTrue(numpy.all(background_subtracted.data > -1))

        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()
