import numpy
import typing
import unittest

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.swift import Facade
from nion.swift.model import DataItem
from nion.swift.model import Graphics
from nion.swift.model import PlugInManager
from nion.swift.test import TestContext

# pytest imports the plug-in as nion_eels_analysis, so an absolute import would load it a second time and register its schema entities twice.
from .. import MenuExtension


Facade.initialize()


class TestEELSMenu(unittest.TestCase):

    def setUp(self) -> None:
        self._test_setup = TestContext.TestSetup(set_global=True)

    def tearDown(self) -> None:
        self._test_setup = typing.cast(typing.Any, None)

    def test_fit_background_chosen_from_eels_menu_adds_background_and_signal_to_spectrum(self) -> None:
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data = numpy.random.uniform(10, 1000, 1024).astype(numpy.float32)
            xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=Calibration.Calibration(units="~"), dimensional_calibrations=[Calibration.Calibration(scale=2.0, units="eV")])
            data_item = DataItem.new_data_item(xdata)
            document_model.append_data_item(data_item)
            display_item = document_model.get_display_item_for_data_item(data_item)
            assert display_item
            interval = Graphics.IntervalGraphic()
            interval.start = 0.2
            interval.end = 0.3
            display_item.add_graphic(interval)
            display_item.graphic_selection.set(0)
            document_controller.selected_display_panel.set_display_panel_display_item(display_item)
            # the extension builds the EELS menu in every window, including this one, when it is made.
            menu_extension = MenuExtension(PlugInManager.APIBroker())
            try:
                eels_menu = document_controller.get_or_create_menu("eels_menu", "EELS", "window_menu")
                fit_background_menu_action = next(menu_action for menu_action in eels_menu.get_menu_actions() if menu_action.title == "Fit Background")
                fit_background_menu_action.trigger()
                document_model.recompute_all()
                document_controller.periodic()
                # the spectrum's display shows the fitted background and the subtracted signal along with the spectrum.
                self.assertEqual(3, len(display_item.data_items))
            finally:
                menu_extension.close()

    def test_measure_temperature_chosen_from_eels_menu_with_one_spectrum_selected_explains_and_adds_nothing(self) -> None:
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data = numpy.random.uniform(10, 1000, 1024).astype(numpy.float32)
            xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=Calibration.Calibration(units="~"), dimensional_calibrations=[Calibration.Calibration(scale=0.001, units="eV")])
            data_item = DataItem.new_data_item(xdata)
            document_model.append_data_item(data_item)
            display_item = document_model.get_display_item_for_data_item(data_item)
            assert display_item
            # the selection comes from the data panel when no display panel is selected.
            document_controller.selected_display_panel = None
            menu_extension = MenuExtension(PlugInManager.APIBroker())
            try:
                eels_menu = document_controller.get_or_create_menu("eels_menu", "EELS", "window_menu")
                measure_temperature_menu_action = next(menu_action for menu_action in eels_menu.get_menu_actions() if menu_action.title == "Measure Temperature")
                document_controller.select_display_items_in_data_panel([display_item])
                measure_temperature_menu_action.trigger()
                # the instructions are in a dialog, which is closed here since the test context doesn't close it.
                dialogs = [window for window in document_controller.app.windows if window is not document_controller]
                for dialog in dialogs:
                    dialog.request_close()
                self.assertEqual(1, len(document_model.data_items), "Measure Temperature with one spectrum selected compares it with itself")
                self.assertEqual(1, len(dialogs), "Measure Temperature with one spectrum selected doesn't explain how to use it")
            finally:
                menu_extension.close()

    def test_fit_background_chosen_from_eels_menu_without_an_interval_selected_explains_and_adds_nothing(self) -> None:
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data = numpy.random.uniform(10, 1000, 1024).astype(numpy.float32)
            xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=Calibration.Calibration(units="~"), dimensional_calibrations=[Calibration.Calibration(scale=2.0, units="eV")])
            data_item = DataItem.new_data_item(xdata)
            document_model.append_data_item(data_item)
            display_item = document_model.get_display_item_for_data_item(data_item)
            assert display_item
            document_controller.selected_display_panel.set_display_panel_display_item(display_item)
            menu_extension = MenuExtension(PlugInManager.APIBroker())
            try:
                eels_menu = document_controller.get_or_create_menu("eels_menu", "EELS", "window_menu")
                fit_background_menu_action = next(menu_action for menu_action in eels_menu.get_menu_actions() if menu_action.title == "Fit Background")
                fit_background_menu_action.trigger()
                # the instructions are in a dialog, which is closed here since the test context doesn't close it.
                dialogs = [window for window in document_controller.app.windows if window is not document_controller]
                for dialog in dialogs:
                    dialog.request_close()
                self.assertEqual(1, len(document_model.data_items), "Fit Background without an interval selected adds a background")
                self.assertEqual(1, len(dialogs), "Fit Background without an interval selected doesn't explain how to use it")
            finally:
                menu_extension.close()

    def test_map_signal_chosen_from_eels_menu_with_an_interval_on_a_plain_spectrum_explains_and_adds_nothing(self) -> None:
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data = numpy.random.uniform(10, 1000, 1024).astype(numpy.float32)
            xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=Calibration.Calibration(units="~"), dimensional_calibrations=[Calibration.Calibration(scale=2.0, units="eV")])
            data_item = DataItem.new_data_item(xdata)
            document_model.append_data_item(data_item)
            display_item = document_model.get_display_item_for_data_item(data_item)
            assert display_item
            interval = Graphics.IntervalGraphic()
            interval.start = 0.2
            interval.end = 0.3
            display_item.add_graphic(interval)
            display_item.graphic_selection.set(0)
            document_controller.selected_display_panel.set_display_panel_display_item(display_item)
            menu_extension = MenuExtension(PlugInManager.APIBroker())
            try:
                eels_menu = document_controller.get_or_create_menu("eels_menu", "EELS", "window_menu")
                map_signal_menu_action = next(menu_action for menu_action in eels_menu.get_menu_actions() if menu_action.title == "Map Signal")
                map_signal_menu_action.trigger()
                # the instructions are in a dialog, which is closed here since the test context doesn't close it.
                dialogs = [window for window in document_controller.app.windows if window is not document_controller]
                for dialog in dialogs:
                    dialog.request_close()
                self.assertEqual(1, len(document_model.data_items), "Map Signal on a spectrum which isn't background subtracted adds a map")
                self.assertEqual(1, len(dialogs), "Map Signal on a spectrum which isn't background subtracted doesn't explain how to use it")
            finally:
                menu_extension.close()

    def test_map_thickness_chosen_from_eels_menu_with_a_spectrum_selected_explains_and_adds_nothing(self) -> None:
        with TestContext.create_memory_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data = numpy.random.uniform(10, 1000, 1024).astype(numpy.float32)
            xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=Calibration.Calibration(units="~"), dimensional_calibrations=[Calibration.Calibration(scale=2.0, units="eV")])
            data_item = DataItem.new_data_item(xdata)
            document_model.append_data_item(data_item)
            display_item = document_model.get_display_item_for_data_item(data_item)
            assert display_item
            document_controller.selected_display_panel.set_display_panel_display_item(display_item)
            menu_extension = MenuExtension(PlugInManager.APIBroker())
            try:
                eels_menu = document_controller.get_or_create_menu("eels_menu", "EELS", "window_menu")
                map_thickness_menu_action = next(menu_action for menu_action in eels_menu.get_menu_actions() if menu_action.title == "Map Thickness")
                map_thickness_menu_action.trigger()
                # the instructions are in a dialog, which is closed here since the test context doesn't close it.
                dialogs = [window for window in document_controller.app.windows if window is not document_controller]
                for dialog in dialogs:
                    dialog.request_close()
                self.assertEqual(1, len(document_model.data_items), "Map Thickness with a spectrum selected maps it to a single value")
                self.assertEqual(1, len(dialogs), "Map Thickness with a spectrum selected doesn't explain how to use it")
            finally:
                menu_extension.close()
