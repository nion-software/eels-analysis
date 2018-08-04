# run this from the command line using:
# cd EELSAnalysis
# python -m unittest test/Library_test.py

import numpy
import os
import random
import sys
import unittest

# niondata must be available as a module.
# it can be added using something similar to
#   conda dev /path/to/niondata
from nion.data import Calibration
from nion.data import DataAndMetadata

sys.path.append(os.path.dirname(os.path.realpath(os.path.join(__file__, "..", ".."))))

# Note: EELSAnalysis is only available in sys.path above is appended with its _parent_ directory.
from nion.eels_analysis import PeriodicTable
from nion.eels_analysis import eels_analysis


class TestEELSAnalysisFunctions(unittest.TestCase):

    def setUp(self):
        """Common code for all tests can go here."""
        pass

    def tearDown(self):
        """Common code for all tests can go here."""
        pass

    def test_slow_linear_background(self):
        height, width, depth = 4, 5, 6
        data = numpy.zeros((height, width, depth))
        for row in range(height):
            for column in range(width):
                r1 = random.randint(0, 100)
                r2 = random.randint(0, 100)
                data[row, column, :] = numpy.linspace(r1, r2, depth) + numpy.random.uniform(-1, 1, depth) / 2
        linear_background = eels_analysis.slow_linear_background(data, -1)
        linear_background_subtracted = data - linear_background
        self.assertEqual(linear_background_subtracted.shape, (height, width, depth))
        self.assertTrue(numpy.all(linear_background_subtracted < 1))
        self.assertTrue(numpy.all(linear_background_subtracted > -1))

    def test_stacked_linear_background(self):
        height, width, depth = 4, 5, 6
        data = numpy.zeros((height, width, depth))
        for row in range(height):
            for column in range(width):
                r1 = random.randint(0, 100)
                r2 = random.randint(0, 100)
                data[row, column, :] = numpy.linspace(r1, r2, depth) + numpy.random.uniform(-1, 1, depth) / 2
        linear_background = eels_analysis.stacked_linear_background(data, -1)
        linear_background_subtracted = data - linear_background
        self.assertEqual(linear_background_subtracted.shape, (height, width, depth))
        self.assertTrue(numpy.all(linear_background_subtracted < 1))
        self.assertTrue(numpy.all(linear_background_subtracted > -1))

    def test_subtract_linear_background(self):
        height, width, depth = 6, 8, 80
        data = numpy.zeros((height, width, depth))
        for row in range(height):
            for column in range(width):
                r1 = random.randint(0, 100)
                r2 = random.randint(0, 100)
                data[row, column, :] = numpy.linspace(r1, r2, depth) + numpy.random.uniform(-1, 1, depth) / 2
        data_and_metadata = DataAndMetadata.DataAndMetadata.from_data(data)
        background_subtracted = eels_analysis.subtract_linear_background(data_and_metadata, (0.0, 1.0), (0.0, 1.0))
        self.assertEqual(background_subtracted.data_shape, (height, width, depth), -1)
        self.assertTrue(numpy.all(numpy.less(background_subtracted.data, 1)))
        self.assertTrue(numpy.all(numpy.greater(background_subtracted.data, -1)))

    def test_subtract_linear_background_with_different_signal_and_fit(self):
        height, width, depth = 6, 8, 80
        data = numpy.zeros((height, width, depth))
        for row in range(height):
            for column in range(width):
                r1 = random.randint(0, 100)
                r2 = random.randint(0, 100)
                data[row, column, :] = numpy.linspace(r1, r2, depth) + numpy.random.uniform(-1, 1, depth) / 2
        data_and_metadata = DataAndMetadata.DataAndMetadata.from_data(data)
        background_subtracted = eels_analysis.subtract_linear_background(data_and_metadata, (0.1, 0.5), (0.6, 0.9))
        self.assertEqual(background_subtracted.data_shape, (height, width, int(depth * 0.3)))
        self.assertTrue(numpy.all(numpy.less(background_subtracted.data[..., 54:72], 1)))
        self.assertTrue(numpy.all(numpy.greater(background_subtracted.data[..., 54:72], -1)))

    def test_signal_and_background_shape_are_consistent_1d(self):
        calibration = Calibration.Calibration(418.92, 0.97, 'eV')
        data_and_metadata = DataAndMetadata.DataAndMetadata.from_data(numpy.ones((2048, ), numpy.float), dimensional_calibrations=[calibration])
        fit_range = 0.2, 0.3
        signal_range = 0.4, 0.5
        signal = eels_analysis.extract_original_signal(data_and_metadata, [fit_range], signal_range)
        background = eels_analysis.calculate_background_signal(data_and_metadata, [fit_range], signal_range)
        self.assertEqual(signal.data_shape, signal.data.shape)
        self.assertEqual(background.data_shape, background.data.shape)

    def test_extracted_signal_has_correct_calibration_and_data(self):
        calibration = Calibration.Calibration(200.0, 2.0, 'eV')
        spectrum_length = 1000
        data_and_metadata = DataAndMetadata.DataAndMetadata.from_data((numpy.random.randn(spectrum_length) * 100).astype(numpy.int32), dimensional_calibrations=[calibration])
        signal = eels_analysis.extract_original_signal(data_and_metadata, [(0.2, 0.3)], (0.4, 0.5))
        self.assertEqual(data_and_metadata.dimensional_calibrations[0], calibration)  # dummy check
        self.assertAlmostEqual(signal.dimensional_calibrations[0].offset, 0.2 * spectrum_length * calibration.scale + calibration.offset)
        self.assertAlmostEqual(signal.dimensional_calibrations[0].scale, calibration.scale)
        self.assertEqual(signal.dimensional_calibrations[0].units, calibration.units)
        self.assertTrue(numpy.array_equal(signal.data, data_and_metadata.data[200:500]))

    def test_background_signal_has_correct_calibration_offset(self):
        calibration = Calibration.Calibration(200.0, 2.0, 'eV')
        spectrum_length = 1000
        data_and_metadata = DataAndMetadata.DataAndMetadata.from_data(numpy.ones((spectrum_length,), numpy.float), dimensional_calibrations=[calibration])
        background = eels_analysis.calculate_background_signal(data_and_metadata, [(0.2, 0.3)], (0.4, 0.5))
        self.assertEqual(data_and_metadata.dimensional_calibrations[0], calibration)  # dummy check
        self.assertAlmostEqual(background.dimensional_calibrations[0].offset, 0.2 * spectrum_length * calibration.scale + calibration.offset)
        self.assertAlmostEqual(background.dimensional_calibrations[0].scale, calibration.scale)
        self.assertEqual(background.dimensional_calibrations[0].units, calibration.units)

    def test_make_signal_like_puts_data_in_correct_place(self):
        calibration = Calibration.Calibration(200.0, 2.0, 'eV')
        spectrum_length = 1000
        data_and_metadata = DataAndMetadata.DataAndMetadata.from_data(numpy.ones((spectrum_length,), numpy.float), dimensional_calibrations=[calibration])
        signal = eels_analysis.extract_original_signal(data_and_metadata, [(0.2, 0.3)], (0.4, 0.5))
        signal = DataAndMetadata.DataAndMetadata.from_data(numpy.ones(300, ), signal.intensity_calibration, signal.dimensional_calibrations)
        expanded = eels_analysis.make_signal_like(signal, data_and_metadata)
        self.assertEqual(expanded.dimensional_calibrations[0], calibration)
        self.assertTrue(numpy.array_equal(expanded.data[0:200], numpy.zeros((200, ))))
        self.assertTrue(numpy.array_equal(expanded.data[200:500], numpy.ones((300, ))))
        self.assertTrue(numpy.array_equal(expanded.data[500:1000], numpy.zeros((500, ))))

    def test_map_background_subtracted_signal_produces_correct_calibrations(self):
        calibration = Calibration.Calibration(200.0, 2.0, 'eV')
        calibration_y = Calibration.Calibration(101.0, 1.5, 'nm')
        calibration_x = Calibration.Calibration(102.0, 2.5, 'nm')
        spectrum_length = 1000
        w, h = 20, 20
        electron_shell = PeriodicTable.ElectronShell(1, 1, 0)
        data_and_metadata = DataAndMetadata.DataAndMetadata.from_data(numpy.ones((spectrum_length, w, h), numpy.float), dimensional_calibrations=[calibration_y, calibration_x, calibration])
        mapped = eels_analysis.map_background_subtracted_signal(data_and_metadata, electron_shell, [(0.2, 0.3)], (0.4, 0.5))
        self.assertEqual(len(mapped.dimensional_shape), 2)
        self.assertEqual(len(mapped.dimensional_calibrations), 2)
        self.assertEqual(mapped.dimensional_calibrations[0], calibration_y)
        self.assertEqual(mapped.dimensional_calibrations[1], calibration_x)


if __name__ == '__main__':
    unittest.main()
