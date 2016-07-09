# run this from the command line using:
# cd EELSAnalysis
# python -m unittest test/core_loss_edge_test.py

import os
import sys
import unittest

import numpy
import scipy.stats

sys.path.append(os.path.dirname(os.path.realpath(os.path.join(__file__, "..", ".."))))

from EELSAnalysis import EELS_DataAnalysis as analyzer

"""
linear = linspace(0,1,1024)
20 * square(1 - linear) + rescale(gammapdf(linear, 1.3, 0.4, 0.01), (0, 3))
"""
class TestLibrary(unittest.TestCase):

    def setUp(self):
        """Common code for all tests can go here."""
        pass

    def tearDown(self):
        """Common code for all tests can go here."""
        pass

    def test_power_law_background_1d(self):
        # this edge in Swift:
        # linear = linspace(0,1,1000)
        # 20 * pow(1 - linear, 4)
        linear = numpy.linspace(0, 1, 1000)
        background = 20 * numpy.power(numpy.subtract(1, linear), 4)
        spectral_range = numpy.array([0, 1000])
        edge_onset = 500.0
        edge_delta = 100.0
        bkgd_range = numpy.array([400.0, 500.0])
        edge_map, edge_profile, bkgd_model, profile_range = analyzer.core_loss_edge(background, spectral_range, edge_onset, edge_delta, bkgd_range)
        self.assertLess(abs(numpy.amin(bkgd_model) - numpy.amin(background[400:600])), numpy.ptp(background[400:600]) / 10.0)
        self.assertLess(abs(numpy.amax(bkgd_model) - numpy.amax(background[400:600])), numpy.ptp(background[400:600]) / 10.0)
        self.assertLess(numpy.average(bkgd_model - background[400:600]), 0.1)


    def test_core_loss_edge_1d(self):
        # this edge in Swift:
        # linear = linspace(0,1,1000)
        # 20 * pow(1 - linear, 4) + rescale(gammapdf(linear, 1.3, 0.4, 0.03), (0, 3))
        linear = numpy.linspace(0, 1, 1000)
        background = 20 * numpy.power(numpy.subtract(1, linear), 4)
        raw_signal = scipy.stats.gamma(a=1.3, loc=0.4, scale=0.03).pdf(linear)
        signal = 3.0 * (raw_signal - numpy.amin(raw_signal)) / numpy.ptp(raw_signal)
        spectrum = background + signal
        spectral_range = numpy.array([300, 2300])
        edge_onset = 1100.0
        edge_delta = 200.0
        bkgd_range = numpy.array([950.0, 1050.0])
        edge_map, edge_profile, bkgd_model, profile_range = analyzer.core_loss_edge(spectrum, spectral_range, edge_onset, edge_delta, bkgd_range)
        self.assertEqual(edge_map.shape, (1, ))
        # print(numpy.sum(signal))
        # print(numpy.sum(edge_profile))
        # print(numpy.sum(spectrum[325:500]))
        # print('t', numpy.trapz(edge_profile, dx=2.0))
        # print('s', numpy.trapz(signal[325:500], dx=2.0))
        # self.assertAlmostEqual(edge_map[0], 64.2302, 3)
        self.assertEqual(edge_profile.shape, (1, 175))
        self.assertLess(abs(numpy.amax(signal) - numpy.amax(edge_profile)), 0.3)
        # print('signal size', signal[325:500].shape)
        # print('signal min', numpy.amin(signal[325:500]))
        # print('signal max', numpy.amax(signal[325:500]))
        # print('edge size', edge_profile.shape)
        # print('edge min', numpy.amin(edge_profile))
        # print('edge max', numpy.amax(edge_profile))
        # print(bkgd_model.shape)
        # print(numpy.amax(bkgd_model))
        # print(numpy.amin(bkgd_model))
        # print(profile_range)
