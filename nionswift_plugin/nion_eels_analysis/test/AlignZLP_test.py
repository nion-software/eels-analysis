import typing
import unittest

import numpy
import scipy

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.swift import Facade
from nion.swift.model import DataItem
from nion.swift.test import TestContext

from .. import AlignZLP


Facade.initialize()


def create_memory_profile_context() -> TestContext.MemoryProfileContext:
    return TestContext.MemoryProfileContext()


def generate_peak_data(*, range_ev: float = 100.0, length: int = 1000, add_noise: bool = False, is_biased: bool = False) -> DataAndMetadata.DataAndMetadata:
    x_axis = numpy.arange(-range_ev / 10, range_ev, range_ev / length)
    if is_biased:
        x_axis[length // 10:] = numpy.arange(0, range_ev / 2.5, range_ev / 2.5 / length)
    data = 1e6 * scipy.stats.norm.pdf(x_axis, 0, 1)
    if add_noise:
        data += numpy.abs(numpy.random.normal(0, 5, data.shape))
    intensity_calibration = Calibration.Calibration(units="counts")
    dimensional_calibrations = [Calibration.Calibration(scale=range_ev / length, offset=-range_ev / 10, units="eV")]
    return DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)


class TestBackgroundSubtraction(unittest.TestCase):

    def setUp(self) -> None:
        self._test_setup = TestContext.TestSetup(set_global=True)

    def tearDown(self) -> None:
        self._test_setup = typing.cast(typing.Any, None)

    def test_calibrate_spectrum_for_single_spectrum(self) -> None:
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
            dialog = AlignZLP._calibrate_spectrum(api, api.application.document_windows[0])
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

    def test_align_zlp_computation(self) -> None:
        with create_memory_profile_context() as test_context:
            document_controller = test_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            peak_xdata = generate_peak_data()
            eels_data_item = DataItem.new_data_item(peak_xdata)
            document_model.append_data_item(eels_data_item)
            display_panel = document_controller.selected_display_panel
            display_item = document_model.get_display_item_for_data_item(eels_data_item)
            display_panel.set_display_panel_display_item(display_item)
            api = Facade.get_api("~1.0", "~1.0")
            AlignZLP.apply_align_zlp(api, Facade.DocumentWindow(document_controller))
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            self.assertEqual(2, len(document_model.data_items))
            self.assertIn("(Align ZLP)", document_model.data_items[1].title)
            self.assertAlmostEqual(50.0, numpy.argmax(document_model.data_items[1].xdata))
            self.assertAlmostEqual(0.0, document_model.data_items[1].dimensional_calibrations[-1].convert_to_calibrated_value(50.5))
