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

from nion.eels_analysis import EELS_DataAnalysis as analyzer


class TestLibrary(unittest.TestCase):

    def setUp(self):
        """Common code for all tests can go here."""
        pass

    def tearDown(self):
        """Common code for all tests can go here."""
        pass

    def test_power_law_background_1d(self):
        # this edge in Swift:
        # 10 * pow(linspace(1,10,1000), -4)
        scale = 1E4
        background = scale * numpy.power(numpy.linspace(1,10,1000), -4)
        self.assertGreaterEqual(numpy.amin(background), 1)
        spectral_range = numpy.array([0, 1000])
        edge_onset = 500.0
        edge_delta = 100.0
        bkgd_range = numpy.array([400.0, 500.0])
        signal_background = background[400:600]
        edge_map, edge_profile, bkgd_model, profile_range = analyzer.core_loss_edge(background, spectral_range, edge_onset, edge_delta, bkgd_range)
        self.assertEqual(bkgd_model.shape, (1, 200))
        self.assertLess(abs(numpy.amin(bkgd_model) - numpy.amin(signal_background)), numpy.ptp(signal_background) / 100.0)
        self.assertLess(abs(numpy.amax(bkgd_model) - numpy.amax(signal_background)), numpy.ptp(signal_background) / 100.0)
        self.assertLess(numpy.average(bkgd_model - signal_background), scale * 0.0001)

    def test_power_law_background_1d_with_negative_values(self):
        # tests only that it can handle negative values; not that it is accurate
        shape = 1000
        background = 10000 * numpy.power(numpy.linspace(1,10,shape), -4)  # 0 -> 10000
        background = background + numpy.random.poisson(numpy.full((shape,), 100)) - 100
        fit_range = range(400, 500)
        self.assertLess(numpy.amin(background[fit_range]), 0)
        spectral_range = numpy.array([0, shape])
        edge_onset = fit_range.stop
        edge_delta = 100.0
        bkgd_range = numpy.array([fit_range.start, fit_range.stop])
        analyzer.core_loss_edge(background, spectral_range, edge_onset, edge_delta, bkgd_range)

    def test_core_loss_edge_1d(self):
        # this edge in Swift:
        # 10 * pow(linspace(1,10,1000), -4) + 0.01 * rescale(gammapdf(linspace(0, 1, 1000), 1.3, 0.5, 0.01))
        scale = 1E4
        background = scale * numpy.power(numpy.linspace(1,10,1000), -4)
        raw_signal = scipy.stats.gamma(a=1.3, loc=0.5, scale=0.01).pdf(numpy.linspace(0,1,1000))
        signal = scale / 1000 * (raw_signal - numpy.amin(raw_signal)) / numpy.ptp(raw_signal)
        spectrum = background + signal
        self.assertGreaterEqual(numpy.amin(spectrum), 1)
        spectral_range = numpy.array([0, 1000])
        edge_onset = 500.0
        edge_delta = 100.0
        bkgd_range = numpy.array([400.0, 500.0])
        edge_map, edge_profile, bkgd_model, profile_range = analyzer.core_loss_edge(spectrum, spectral_range, edge_onset, edge_delta, bkgd_range)
        self.assertEqual(tuple(profile_range), (400, 600))
        self.assertEqual(edge_map.shape, (1, ))
        signal_slice = signal[400:600]
        expected_edge_map = numpy.trapz(signal_slice)
        log10_scale = int(-math.log10(scale) + 1)  # 1/10 of scale
        self.assertAlmostEqual(edge_map[0], expected_edge_map, 1 + log10_scale)  # within 1/10
        self.assertEqual(edge_profile.shape, (1, 200))
        self.assertAlmostEqual(numpy.amin(signal_slice), numpy.amin(edge_profile), 2 + log10_scale)  # within 1/100
        self.assertAlmostEqual(numpy.amax(signal_slice), numpy.amax(edge_profile), 2 + log10_scale)  # within 1/100
        self.assertAlmostEqual(numpy.average(signal_slice), numpy.average(edge_profile), 4 + log10_scale)  # within 1/10000

if __name__ == '__main__':
    unittest.main()
