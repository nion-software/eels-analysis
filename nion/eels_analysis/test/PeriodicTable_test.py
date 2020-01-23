# run this from the command line using:
# cd EELSAnalysis
# python -m unittest test/core_loss_edge_test.py

import fractions
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.realpath(os.path.join(__file__, "..", ".."))))

from nion.eels_analysis import PeriodicTable


class TestLibrary(unittest.TestCase):

    def setUp(self):
        """Common code for all tests can go here."""
        pass

    def tearDown(self):
        """Common code for all tests can go here."""
        pass

    def test_subshell_label(self):
        # Test all possible labels up to f-states
        subshell_labels = [None, "s", "p", "p", "d", "d", "f", "f"]
        spin_numerators = [None, 1, 1, 3, 3, 5, 5, 7]
        for subshell_index in range(len(subshell_labels))[1:]:
            electron_shell = PeriodicTable.ElectronShell(99, 4, subshell_index)
            self.assertEqual(electron_shell.subshell_label, subshell_labels[subshell_index])
            self.assertEqual(electron_shell.spin_fraction, fractions.Fraction(spin_numerators[subshell_index], 2))

if __name__ == '__main__':
    unittest.main()
