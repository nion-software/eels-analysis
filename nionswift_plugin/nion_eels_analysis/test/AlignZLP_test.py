import numpy
import unittest

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.swift import Application
from nion.swift import Facade
from nion.swift.model import DataItem
from nion.swift.test import TestContext
from nion.ui import TestUI

from nionswift_plugin.nion_eels_analysis import AlignZLP


Facade.initialize()


class TestBackgroundSubtraction(unittest.TestCase):

    def setUp(self):
        self.app = Application.Application(TestUI.UserInterface(), set_global=True)

    def tearDown(self):
        pass

    def test_calibrate_spectrum_for_single_spectrum(self):
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data = numpy.zeros((100,), dtype=numpy.float32)
            data[10] = 1
            intensity_calibration = Calibration.Calibration(units="~")
            dimensional_calibrations = [Calibration.Calibration(scale=1.0, units="eV")]
            data_descriptor = DataAndMetadata.DataDescriptor(is_sequence=False, collection_dimension_count=0, datum_dimension_count=1)
            xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)
            data_item = DataItem.new_data_item(xdata)
            document_model.append_data_item(data_item)
            display_item = document_model.get_display_item_for_data_item(data_item)
            document_controller.select_display_items_in_data_panel([display_item])
            document_controller.data_panel_focused()
            api = Facade.get_api("~1.0", "~1.0")
            dialog = AlignZLP.calibrate_spectrum(api, api.application.document_windows[0])
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(api.library.data_items))
            # calibrate_spectrum should create two graphics, offset and scale
            self.assertEqual(2, len(api.library.data_items[0].graphics))
            # It should find the peak at 10 and set the calibration offset accordingly
            self.assertAlmostEqual(data_item.dimensional_calibrations[0].offset, -10.5)
            # Move the scale graphic and check that the calibration changed accordingly
            offset_graphic = api.library.data_items[0].graphics[0]
            scale_graphic = api.library.data_items[0].graphics[1]
            position_diff = scale_graphic.position - offset_graphic.position
            scale_graphic.position = offset_graphic.position + 0.5 * position_diff
            self.assertAlmostEqual(data_item.dimensional_calibrations[0].scale, 2.0)
            # Move the offset graphic and check that the calibration offset and the scale graphic have moved
            position_diff = scale_graphic.position - offset_graphic.position
            offset_graphic.position += 0.1
            self.assertAlmostEqual(position_diff, scale_graphic.position - offset_graphic.position)
            self.assertAlmostEqual(data_item.dimensional_calibrations[0].offset, -41)
            # Closing the dialog should remove the graphics
            dialog.request_close()
            self.assertEqual(0, len(api.library.data_items[0].graphics))
            # Test cleanup
            document_model.remove_data_item(data_item)
            self.assertEqual(0, len(document_model.data_items))
            self.assertEqual(0, len(document_model.display_items))
            self.assertEqual(0, len(document_model.data_structures))
