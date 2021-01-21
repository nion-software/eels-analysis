import copy
import numpy
import unittest

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.swift import Application
from nion.swift import Facade
from nion.swift.model import DataItem
from nion.swift.model import Graphics
from nion.swift.test import TestContext
from nion.ui import TestUI

from nionswift_plugin.nion_eels_analysis import PeakFitting


Facade.initialize()


class TestBackgroundSubtraction(unittest.TestCase):

    def setUp(self):
        self.app = Application.Application(TestUI.UserInterface(), set_global=True)

    def tearDown(self):
        pass

    def __create_spectrum(self) -> DataItem.DataItem:
        data = numpy.random.uniform(10, 1000, 1024).astype(numpy.float32)
        intensity_calibration = Calibration.Calibration(units="~")
        dimensional_calibrations = [Calibration.Calibration(scale=2.0, units="eV")]
        data_descriptor = DataAndMetadata.DataDescriptor(is_sequence=False, collection_dimension_count=0, datum_dimension_count=1)
        xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)
        return DataItem.new_data_item(xdata)

    def test_add_peak_fitting_computation_and_remove(self):
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data_item = self.__create_spectrum()
            document_model.append_data_item(data_item)
            display_item = document_model.get_display_item_for_data_item(data_item)
            api = Facade.get_api("~1.0", "~1.0")
            library = api.library
            api_data_item = library.data_items[0]
            api_display_item = api_data_item.display
            PeakFitting.add_peak_fitting_computation(api, library, api_display_item, api_data_item)
            self.assertEqual(3, len(document_model.data_items))
            self.assertEqual(3, len(document_model.display_items))
            self.assertEqual(3, len(display_item.data_items))
            self.assertEqual(3, len(display_item.display_layers))
            self.assertEqual(1, len(document_model.data_structures))
            self.assertEqual(1, len(document_model.computations))
            document_model.remove_data_item(data_item)
            self.assertEqual(0, len(document_model.data_items))
            self.assertEqual(0, len(document_model.display_items))
            self.assertEqual(0, len(document_model.data_structures))
            self.assertEqual(0, len(document_model.computations))

    def test_peak_fitting_computation_is_removed_when_peak_removed(self):
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data_item = self.__create_spectrum()
            document_model.append_data_item(data_item)
            display_item = document_model.get_display_item_for_data_item(data_item)
            api = Facade.get_api("~1.0", "~1.0")
            library = api.library
            api_data_item = library.data_items[0]
            api_display_item = api_data_item.display
            PeakFitting.add_peak_fitting_computation(api, library, api_display_item, api_data_item)
            self.assertEqual(3, len(document_model.data_items))
            self.assertEqual(3, len(document_model.display_items))
            self.assertEqual(3, len(display_item.data_items))
            self.assertEqual(3, len(display_item.display_layers))
            self.assertEqual(1, len(document_model.data_structures))
            self.assertEqual(1, len(document_model.computations))
            document_model.remove_data_item(document_model.data_items[-2])
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(0, len(document_model.data_structures))
            self.assertEqual(0, len(document_model.computations))

    def test_peak_fitting_computation_is_removed_when_subtracted_removed(self):
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data_item = self.__create_spectrum()
            document_model.append_data_item(data_item)
            display_item = document_model.get_display_item_for_data_item(data_item)
            api = Facade.get_api("~1.0", "~1.0")
            library = api.library
            api_data_item = library.data_items[0]
            api_display_item = api_data_item.display
            PeakFitting.add_peak_fitting_computation(api, library, api_display_item, api_data_item)
            self.assertEqual(3, len(document_model.data_items))
            self.assertEqual(3, len(document_model.display_items))
            self.assertEqual(3, len(display_item.data_items))
            self.assertEqual(3, len(display_item.display_layers))
            self.assertEqual(1, len(document_model.data_structures))
            self.assertEqual(1, len(document_model.computations))
            document_model.remove_data_item(document_model.data_items[-1])
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(0, len(document_model.data_structures))
            self.assertEqual(0, len(document_model.computations))

    def test_peak_fitting_computation_is_removed_when_all_peak_displays_removed(self):
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data_item = self.__create_spectrum()
            document_model.append_data_item(data_item)
            display_item = document_model.get_display_item_for_data_item(data_item)
            api = Facade.get_api("~1.0", "~1.0")
            library = api.library
            api_data_item = library.data_items[0]
            api_display_item = api_data_item.display
            PeakFitting.add_peak_fitting_computation(api, library, api_display_item, api_data_item)
            self.assertEqual(3, len(document_model.data_items))
            self.assertEqual(3, len(document_model.display_items))
            self.assertEqual(3, len(display_item.data_items))
            self.assertEqual(3, len(display_item.display_layers))
            self.assertEqual(1, len(document_model.data_structures))
            self.assertEqual(1, len(document_model.computations))
            document_model.remove_display_item(document_model.display_items[-2])
            document_model.remove_display_item(document_model.display_items[-1])
            document_model.display_items[0].remove_display_data_channel(document_model.display_items[0].display_data_channels[-1])
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(0, len(document_model.data_structures))
            self.assertEqual(0, len(document_model.computations))


if __name__ == '__main__':
    unittest.main()
