# run this from the command line using:
# cd EELSAnalysis
# python -m unittest test/core_loss_edge_test.py

import math
import os
import sys
import unittest

import numpy
import scipy.stats

sys.path.append(os.path.dirname(os.path.realpath(os.path.join(__file__, "..", ".."))))

from nion.eels_analysis import eels_analysis
from nion.eels_analysis import FEFF_EELS_Service

class TestLibrary(unittest.TestCase):

    def setUp(self):
        """Common code for all tests can go here."""
        pass

    def tearDown(self):
        """Common code for all tests can go here."""
        pass

    def test_feff_cros_sections(self):
        # Try the gold M45 edges 
        cross_section = eels_analysis.partial_cross_section_nm2(atomic_number = 79, shell_number = 4, subshell_index = 7,
                              edge_onset_eV = 84.0, edge_delta_eV = 25.0, beam_energy_eV = 200000.0,
                                  convergence_angle_rad = 0.01, collection_angle_rad = 0.01)
        assert cross_section > 0.0
        
if __name__ == '__main__':
    unittest.main()
