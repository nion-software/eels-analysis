import copy
import numpy
import scipy
import typing
import unittest

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.swift import Application
from nion.swift import Facade
from nion.swift.model import DataItem
from nion.swift.model import Graphics
from nion.swift.test import TestContext
from nion.ui import TestUI

from .. import BackgroundSubtraction


Facade.initialize()


def generate_peak_data(*, range_ev: float = 100.0, length: int = 1000, add_noise: bool = False) -> DataAndMetadata.DataAndMetadata:
    x_axis = numpy.arange(-range_ev / 10, range_ev, range_ev / length)
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

    def __create_spectrum(self) -> DataItem.DataItem:
        data = numpy.random.uniform(10, 1000, 1024).astype(numpy.float32)
        intensity_calibration = Calibration.Calibration(units="~")
        dimensional_calibrations = [Calibration.Calibration(scale=2.0, units="eV")]
        data_descriptor = DataAndMetadata.DataDescriptor(is_sequence=False, collection_dimension_count=0, datum_dimension_count=1)
        xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)
        return DataItem.new_data_item(xdata)

    def test_add_background_subtraction_computation_and_remove(self) -> None:
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data_item = self.__create_spectrum()
            document_model.append_data_item(data_item)
            display_item = document_model.get_display_item_for_data_item(data_item)
            interval = Graphics.IntervalGraphic()
            interval.start = 0.2
            interval.end = 0.3
            display_item.add_graphic(interval)
            api = Facade.get_api("~1.0", "~1.0")
            library = api.library
            api_data_item = library.data_items[0]
            api_display_item = api_data_item.display
            api_intervals = copy.copy(api_display_item.graphics)
            BackgroundSubtraction.add_background_subtraction_computation(api, library,
                                                                         api_display_item, api_data_item,
                                                                         api_intervals)
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            self.assertEqual(3, len(document_model.data_items))
            self.assertEqual(3, len(document_model.display_items))
            self.assertEqual(3, len(display_item.data_items))
            self.assertEqual(3, len(display_item.display_layers))
            self.assertEqual(1, len(document_model.data_structures))
            self.assertEqual(1, len(document_model.computations))
            document_model.remove_data_item(data_item)
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            self.assertEqual(0, len(document_model.data_items))
            self.assertEqual(0, len(document_model.display_items))
            self.assertEqual(0, len(document_model.data_structures))
            self.assertEqual(0, len(document_model.computations))

    def test_background_subtraction_computation_is_removed_when_background_removed(self) -> None:
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data_item = self.__create_spectrum()
            document_model.append_data_item(data_item)
            display_item = document_model.get_display_item_for_data_item(data_item)
            interval = Graphics.IntervalGraphic()
            interval.start = 0.2
            interval.end = 0.3
            display_item.add_graphic(interval)
            api = Facade.get_api("~1.0", "~1.0")
            library = api.library
            api_data_item = library.data_items[0]
            api_display_item = api_data_item.display
            api_intervals = copy.copy(api_display_item.graphics)
            BackgroundSubtraction.add_background_subtraction_computation(api, library,
                                                                         api_display_item, api_data_item,
                                                                         api_intervals)
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            self.assertEqual(3, len(document_model.data_items))
            self.assertEqual(3, len(document_model.display_items))
            self.assertEqual(3, len(display_item.data_items))
            self.assertEqual(3, len(display_item.display_layers))
            self.assertEqual(1, len(document_model.data_structures))
            self.assertEqual(1, len(document_model.computations))
            document_model.remove_data_item(document_model.data_items[-2])
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(0, len(document_model.data_structures))
            self.assertEqual(0, len(document_model.computations))

    def test_background_subtraction_computation_is_removed_when_subtracted_removed(self) -> None:
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data_item = self.__create_spectrum()
            document_model.append_data_item(data_item)
            display_item = document_model.get_display_item_for_data_item(data_item)
            interval = Graphics.IntervalGraphic()
            interval.start = 0.2
            interval.end = 0.3
            display_item.add_graphic(interval)
            api = Facade.get_api("~1.0", "~1.0")
            library = api.library
            api_data_item = library.data_items[0]
            api_display_item = api_data_item.display
            api_intervals = copy.copy(api_display_item.graphics)
            BackgroundSubtraction.add_background_subtraction_computation(api, library,
                                                                         api_display_item, api_data_item,
                                                                         api_intervals)
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            self.assertEqual(3, len(document_model.data_items))
            self.assertEqual(3, len(document_model.display_items))
            self.assertEqual(3, len(display_item.data_items))
            self.assertEqual(3, len(display_item.display_layers))
            self.assertEqual(1, len(document_model.data_structures))
            self.assertEqual(1, len(document_model.computations))
            document_model.remove_data_item(document_model.data_items[-1])
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(0, len(document_model.data_structures))
            self.assertEqual(0, len(document_model.computations))

    def test_background_subtraction_computation_is_removed_when_last_interval_deleted(self) -> None:
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data_item = self.__create_spectrum()
            document_model.append_data_item(data_item)
            display_item = document_model.get_display_item_for_data_item(data_item)
            interval1 = Graphics.IntervalGraphic()
            interval1.start = 0.2
            interval1.end = 0.3
            display_item.add_graphic(interval1)
            interval2 = Graphics.IntervalGraphic()
            interval2.start = 0.4
            interval2.end = 0.5
            display_item.add_graphic(interval2)
            api = Facade.get_api("~1.0", "~1.0")
            library = api.library
            api_data_item = library.data_items[0]
            api_display_item = api_data_item.display
            api_intervals = copy.copy(api_display_item.graphics)
            BackgroundSubtraction.add_background_subtraction_computation(api, library,
                                                                         api_display_item, api_data_item,
                                                                         api_intervals)
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            self.assertEqual(3, len(document_model.data_items))
            self.assertEqual(3, len(document_model.display_items))
            self.assertEqual(3, len(display_item.data_items))
            self.assertEqual(3, len(display_item.display_layers))
            self.assertEqual(1, len(document_model.data_structures))
            self.assertEqual(1, len(document_model.computations))
            document_model.remove_data_item(interval2)
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            self.assertEqual(3, len(document_model.data_items))
            self.assertEqual(3, len(document_model.display_items))
            self.assertEqual(3, len(display_item.data_items))
            self.assertEqual(3, len(display_item.display_layers))
            self.assertEqual(1, len(document_model.data_structures))
            self.assertEqual(1, len(document_model.computations))
            document_model.remove_data_item(interval1)
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(0, len(document_model.data_structures))
            self.assertEqual(0, len(document_model.computations))

    def test_subtraction_computation(self) -> None:
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            peak_xdata = generate_peak_data()
            si_data = numpy.empty((4, 4, peak_xdata.data.shape[0]), dtype=numpy.float32)
            for i in range(si_data.shape[0]):
                for j in range(si_data.shape[1]):
                    si_data[i, j] = generate_peak_data(add_noise=True)
            si_xdata = DataAndMetadata.new_data_and_metadata(
                si_data,
                intensity_calibration=peak_xdata.intensity_calibration,
                dimensional_calibrations=[Calibration.Calibration(), Calibration.Calibration(), peak_xdata.dimensional_calibrations[-1]],
                data_descriptor=DataAndMetadata.DataDescriptor(False, 2, 1)
            )
            si_data_item = DataItem.new_data_item(si_xdata)
            si_data_item.title = "SI"
            document_model.append_data_item(si_data_item)
            si_display_item = document_model.get_display_item_for_data_item(si_data_item)
            data_item = document_model.get_pick_new(si_display_item, si_data_item)
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            display_item = document_model.get_display_item_for_data_item(data_item)
            interval1 = Graphics.IntervalGraphic()
            interval1.start = 0.2
            interval1.end = 0.3
            display_item.add_graphic(interval1)
            interval2 = Graphics.IntervalGraphic()
            interval2.start = 0.4
            interval2.end = 0.5
            display_item.add_graphic(interval2)
            display_panel = document_controller.selected_display_panel
            display_item = document_model.get_display_item_for_data_item(data_item)
            display_panel.set_display_panel_display_item(display_item)
            api = Facade.get_api("~1.0", "~1.0")
            BackgroundSubtraction.add_background_subtraction_computation(api, Facade.Library(document_model),
                                                                         Facade.Display(display_item), Facade.DataItem(data_item),
                                                                         [Facade.Graphic(interval1), Facade.Graphic(interval2)])
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            BackgroundSubtraction.subtract_background(api, Facade.DocumentWindow(document_controller))
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            self.assertEqual(5, len(document_model.data_items))
            self.assertIn("(EELS Subtract Background)", document_model.data_items[4].title)

    def test_signal_map_computation(self) -> None:
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            peak_xdata = generate_peak_data()
            si_data = numpy.empty((4, 4, peak_xdata.data.shape[0]), dtype=numpy.float32)
            for i in range(si_data.shape[0]):
                for j in range(si_data.shape[1]):
                    si_data[i, j] = generate_peak_data(add_noise=True)
            si_xdata = DataAndMetadata.new_data_and_metadata(
                si_data,
                intensity_calibration=peak_xdata.intensity_calibration,
                dimensional_calibrations=[Calibration.Calibration(), Calibration.Calibration(), peak_xdata.dimensional_calibrations[-1]],
                data_descriptor=DataAndMetadata.DataDescriptor(False, 2, 1)
            )
            si_data_item = DataItem.new_data_item(si_xdata)
            document_model.append_data_item(si_data_item)
            si_display_item = document_model.get_display_item_for_data_item(si_data_item)
            data_item = document_model.get_pick_new(si_display_item, si_data_item)
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            display_item = document_model.get_display_item_for_data_item(data_item)
            interval1 = Graphics.IntervalGraphic()
            interval1.start = 0.2
            interval1.end = 0.3
            display_item.add_graphic(interval1)
            interval2 = Graphics.IntervalGraphic()
            interval2.start = 0.4
            interval2.end = 0.5
            display_item.add_graphic(interval2)
            display_panel = document_controller.selected_display_panel
            display_item = document_model.get_display_item_for_data_item(data_item)
            display_panel.set_display_panel_display_item(display_item)
            api = Facade.get_api("~1.0", "~1.0")
            BackgroundSubtraction.add_background_subtraction_computation(api, Facade.Library(document_model),
                                                                         Facade.Display(display_item), Facade.DataItem(data_item),
                                                                         [Facade.Graphic(interval1), Facade.Graphic(interval2)])
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            BackgroundSubtraction.subtract_background(api, Facade.DocumentWindow(document_controller))
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            display_item.graphic_selection.set(2)
            BackgroundSubtraction.use_signal_for_map(api, Facade.DocumentWindow(document_controller))
            document_model.recompute_all()
            document_controller.periodic()
            self.assertFalse(any(computation.error_text for computation in document_model.computations))
            self.assertEqual(6, len(document_model.data_items))
            self.assertIn("(EELS Map Background Subtracted Signal)", document_model.data_items[5].title)


if __name__ == '__main__':
    unittest.main()
