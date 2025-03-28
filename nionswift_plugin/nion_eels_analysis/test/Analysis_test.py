# standard libraries
import time
import typing
import unittest

# third party libraries
import numpy
import scipy

# local libraries
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.data.xdata_1_0 import data_descriptor
from nion.swift import Application
from nion.swift import Facade
from nion.swift.model import DataItem
from nion.swift.model import PlugInManager
from nion.swift.model import Symbolic
from nion.swift.test import TestContext
from nion.ui import TestUI

from nion.eels_analysis import eels_analysis
from nion.eels_analysis import PeriodicTable

from .. import LiveThickness
from .. import LiveZLP
from .. import PeakFitting
from .. import Thermometry
from .. import ThicknessMap


Facade.initialize()


def create_memory_profile_context() -> TestContext.MemoryProfileContext:
    return TestContext.MemoryProfileContext()


def generate_peak_data(*, range_ev: float = 100.0, length: int = 1000, add_noise: bool = False) -> DataAndMetadata.DataAndMetadata:
    x_axis = numpy.arange(-range_ev / 10, range_ev, range_ev / length)
    x_axis[length // 10:] = numpy.arange(0, range_ev / 2.5, range_ev / 2.5 / length)
    data = 1e6 * scipy.stats.norm.pdf(x_axis, 0, 1)
    if add_noise:
        data += numpy.abs(numpy.random.normal(0, 5, data.shape))
    intensity_calibration = Calibration.Calibration(units="counts")
    dimensional_calibrations = [Calibration.Calibration(scale=range_ev / length, offset=-range_ev / 10, units="eV")]
    return DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)


class TestElementalMappingController(unittest.TestCase):

    def setUp(self) -> None:
        self._test_setup = TestContext.TestSetup(set_global=True)

    def tearDown(self) -> None:
        self._test_setup = typing.cast(typing.Any, None)

    def test_live_thickness_computation(self) -> None:
        with create_memory_profile_context() as test_context:
            document_controller = test_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            peak_xdata = generate_peak_data()
            data_item = DataItem.new_data_item(peak_xdata)
            document_model.append_data_item(data_item)
            display_panel = document_controller.selected_display_panel
            display_item = document_model.get_display_item_for_data_item(data_item)
            display_panel.set_display_panel_display_item(display_item)
            api = Facade.get_api("~1.0", "~1.0")
            LiveThickness.register_measure_thickness_process(api)
            LiveThickness.attach_measure_thickness(api, Facade.DocumentWindow(document_controller))
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            self.assertEqual(1, len(display_item.graphics))
            self.assertEqual("0.3146", display_item.graphics[0].label)  # this value is dependent on the peak data
            self.assertLess(peak_xdata.dimensional_calibrations[-1].convert_to_calibrated_value(display_item.graphics[0].start * peak_xdata.data_shape[0]), 0)
            self.assertGreater(peak_xdata.dimensional_calibrations[-1].convert_to_calibrated_value(display_item.graphics[0].end * peak_xdata.data_shape[0]), 0)

    def test_live_zlp_computation(self) -> None:
        with create_memory_profile_context() as test_context:
            document_controller = test_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            peak_xdata = generate_peak_data()
            data_item = DataItem.new_data_item(peak_xdata)
            document_model.append_data_item(data_item)
            display_panel = document_controller.selected_display_panel
            display_item = document_model.get_display_item_for_data_item(data_item)
            display_panel.set_display_panel_display_item(display_item)
            api = Facade.get_api("~1.0", "~1.0")
            LiveZLP.register_measure_zlp_process(api)
            LiveZLP.attach_measure_zlp(api, Facade.DocumentWindow(document_controller))
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            self.assertEqual(1, len(display_item.graphics))
            self.assertLess(peak_xdata.dimensional_calibrations[-1].convert_to_calibrated_value(display_item.graphics[0].start * peak_xdata.data_shape[0]), 0)
            self.assertGreater(peak_xdata.dimensional_calibrations[-1].convert_to_calibrated_value(display_item.graphics[0].end * peak_xdata.data_shape[0]), 0)

    def test_peak_fitting_computation(self) -> None:
        with create_memory_profile_context() as test_context:
            document_controller = test_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            peak_xdata = generate_peak_data()
            data_item = DataItem.new_data_item(peak_xdata)
            document_model.append_data_item(data_item)
            display_panel = document_controller.selected_display_panel
            display_item = document_model.get_display_item_for_data_item(data_item)
            display_panel.set_display_panel_display_item(display_item)
            api = Facade.get_api("~1.0", "~1.0")
            PeakFitting.fit_zero_loss_peak(api, Facade.DocumentWindow(document_controller))
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            self.assertEqual(3, len(document_model.data_items))
            self.assertIn("(Fit ZLP - Background)", document_model.data_items[1].title)
            self.assertIn("(Fit ZLP - Subtracted)", document_model.data_items[2].title)

    def test_measure_temperature_computation(self) -> None:
        with create_memory_profile_context() as test_context:
            document_controller = test_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data_item1 = DataItem.new_data_item(generate_peak_data(range_ev=10, add_noise=True))
            data_item2 = DataItem.new_data_item(generate_peak_data(range_ev=10, add_noise=True))
            data_item1.title = "A"
            data_item2.title = "B"
            document_model.append_data_item(data_item1)
            document_model.append_data_item(data_item2)
            display_item1 = document_model.get_display_item_for_data_item(data_item1)
            display_item2 = document_model.get_display_item_for_data_item(data_item2)
            api = Facade.get_api("~1.0", "~1.0")
            Thermometry.measure_temperature(api, Facade.DocumentWindow(document_controller), display_items=((display_item1, None), (display_item2, None)))
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            self.assertEqual(5, len(document_model.data_items))
            self.assertIn("Near - Far", document_model.data_items[2].title)
            self.assertIn("Gain", document_model.data_items[3].title)
            self.assertIn("Gain Fit", document_model.data_items[4].title)

    def test_thickness_mapping_computation(self) -> None:
        with create_memory_profile_context() as test_context:
            document_controller = test_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            peak_xdata = generate_peak_data()
            si_data = numpy.empty((4, 4, peak_xdata.data.shape[0]), dtype=numpy.float32)
            for i in range(4):
                for j in range(4):
                    si_data[i, j] = generate_peak_data()
            si_xdata = DataAndMetadata.new_data_and_metadata(
                si_data,
                intensity_calibration=peak_xdata.intensity_calibration,
                dimensional_calibrations=[Calibration.Calibration(), Calibration.Calibration(), peak_xdata.dimensional_calibrations[-1]],
                data_descriptor=DataAndMetadata.DataDescriptor(False, 2, 1)
            )
            data_item = DataItem.new_data_item(si_data)
            document_model.append_data_item(data_item)
            display_panel = document_controller.selected_display_panel
            display_item = document_model.get_display_item_for_data_item(data_item)
            display_panel.set_display_panel_display_item(display_item)
            api = Facade.get_api("~1.0", "~1.0")
            ThicknessMap.map_thickness(api, Facade.DocumentWindow(document_controller))
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            self.assertEqual(2, len(document_model.data_items))
            self.assertIn("(Thickness Map)", document_model.data_items[1].title)
            self.assertAlmostEqual(0.3145582, document_model.data_items[1].data[0, 0])  # dependent on peak data
