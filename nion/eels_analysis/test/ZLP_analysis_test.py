import unittest

from nion.eels_analysis import ZLP_analysis


class TestZLPAnalysis(unittest.TestCase):

    def test_estimate_zlp_amplitude_position_width_fit_spline(self):
        max_pos_in = 61.523
        max_height_in = 1e3
        HWHM_in = 8.225
        data = ZLP_analysis.gaussian(np.arange(512.0), max_height_in, max_pos_in, HWHM_in)
        max_height, max_pos, HWHM = ZLP_analysis.estimate_zlp_amplitude_position_width_fit_spline(data)

        self.assertAlmostEqual(max_pos, max_pos_in)
        self.assertAlmostEqual(max_height, max_height_in)
        self.assertAlmostEqual(HWHM_in, HWHM)

    def test_estimate_zlp_amplitude_position_width_com(self):
        max_pos_in = 61.523
        max_height_in = 1e3
        HWHM_in = 8.225
        data = ZLP_analysis.gaussian(np.arange(512.0), max_height_in, max_pos_in, HWHM_in)
        max_height, max_pos, left_pos, right_pos = ZLP_analysis.estimate_zlp_amplitude_position_width_com(data)
        # com is not as accurate, so have to reduces precision
        self.assertAlmostEqual(max_pos, max_pos_in, places=1)
        self.assertAlmostEqual(max_height, max_height_in, delta=2)
        self.assertAlmostEqual(HWHM_in, (right_pos - left_pos)/2, delta=2)

if __name__ == '__main__':
    unittest.main()