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
        shell=PeriodicTable.ElectronShell(99,4,1)
        self.assertEqual(shell.subshell_label,'s')
        shell=PeriodicTable.ElectronShell(99,4,2)
        self.assertEqual(shell.subshell_label,'p')
        shell=PeriodicTable.ElectronShell(99,4,3)
        self.assertEqual(shell.subshell_label,'p')
        shell=PeriodicTable.ElectronShell(99,4,4)
        self.assertEqual(shell.subshell_label,'d')
        shell=PeriodicTable.ElectronShell(99,4,5)
        self.assertEqual(shell.subshell_label,'d')
        shell=PeriodicTable.ElectronShell(99,4,6)
        self.assertEqual(shell.subshell_label,'f')
        shell=PeriodicTable.ElectronShell(99,4,7)
        self.assertEqual(shell.subshell_label,'f')

    def test_spin_fraction(self):
        import fractions
        # Test all possible spin fraction up to f-states
        shell=PeriodicTable.ElectronShell(99,4,1)
        self.assertEqual(shell.spin_fraction,fractions.Fraction(1,2))
        shell=PeriodicTable.ElectronShell(99,4,2)
        self.assertEqual(shell.spin_fraction,fractions.Fraction(1,2))
        shell=PeriodicTable.ElectronShell(99,4,3)
        self.assertEqual(shell.spin_fraction,fractions.Fraction(3,2))
        shell=PeriodicTable.ElectronShell(99,4,4)
        self.assertEqual(shell.spin_fraction,fractions.Fraction(3,2))
        shell=PeriodicTable.ElectronShell(99,4,5)
        self.assertEqual(shell.spin_fraction,fractions.Fraction(5,2))
        shell=PeriodicTable.ElectronShell(99,4,6)
        self.assertEqual(shell.spin_fraction,fractions.Fraction(5,2))
        shell=PeriodicTable.ElectronShell(99,4,7)
        self.assertEqual(shell.spin_fraction,fractions.Fraction(7,2))
        
if __name__ == '__main__':
    unittest.main()

#for edge in edges:
#    print(edge.atomic_number, edge.shell_number, edge.subshell_index,edge.get_shell_str_in_eels_notation(True),ptable.nominal_binding_energy_ev(edge))
