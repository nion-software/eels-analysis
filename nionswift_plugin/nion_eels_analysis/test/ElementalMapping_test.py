# standard libraries
import contextlib
import time
import unittest

# third party libraries
import numpy

# local libraries
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.swift import Application
from nion.swift import Facade
from nion.swift.model import DataItem
from nion.swift.model import Symbolic
from nion.swift.test import TestContext
from nion.ui import TestUI

from nion.eels_analysis import eels_analysis
from nion.eels_analysis import PeriodicTable

from nionswift_plugin.nion_eels_analysis import ElementalMappingController
from nionswift_plugin.nion_eels_analysis import AlignZLP
from nionswift_plugin.nion_eels_analysis import ThicknessMap


Facade.initialize()


class TestElementalMappingController(unittest.TestCase):

    def setUp(self):
        self.app = Application.Application(TestUI.UserInterface(), set_global=False)

    def tearDown(self):
        pass

    def __create_spectrum_image(self) -> DataItem.DataItem:
        return DataItem.new_data_item(self.__create_spectrum_image_xdata())

    def __create_spectrum_image_xdata(self, dtype=numpy.float32):
        data = numpy.zeros((8, 8, 1024), dtype)
        for row in range(data.shape[0]):
            for column in range(data.shape[1]):
                data[row, column, :] = numpy.random.uniform(10, 1000, 1024)
        intensity_calibration = Calibration.Calibration(units="~")
        dimensional_calibrations = [Calibration.Calibration(units="nm"), Calibration.Calibration(units="nm"), Calibration.Calibration(scale=2.0, units="eV")]
        data_descriptor = DataAndMetadata.DataDescriptor(is_sequence=False, collection_dimension_count=2, datum_dimension_count=1)
        xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)
        return xdata

    def __create_spectrum(self) -> DataItem.DataItem:
        data = numpy.random.uniform(10, 1000, 1024).astype(numpy.float32)
        intensity_calibration = Calibration.Calibration(units="~")
        dimensional_calibrations = [Calibration.Calibration(scale=2.0, units="eV")]
        data_descriptor = DataAndMetadata.DataDescriptor(is_sequence=False, collection_dimension_count=0, datum_dimension_count=1)
        xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)
        return DataItem.new_data_item(xdata)

    def __create_edge(self, model_data_item: DataItem.DataItem, electron_shell: PeriodicTable.ElectronShell):
        binding_energy_eV = PeriodicTable.PeriodicTable().nominal_binding_energy_ev(electron_shell)
        signal_interval_eV = binding_energy_eV, binding_energy_eV * 1.10
        fit_interval_eV = binding_energy_eV * 0.93, binding_energy_eV * 0.98
        dimensional_shape = model_data_item.dimensional_shape
        dimensional_calibrations = model_data_item.dimensional_calibrations
        if dimensional_shape is not None and dimensional_calibrations is not None and len(dimensional_calibrations) > 0:
            calibration = dimensional_calibrations[-1]
            if calibration.units == "eV":
                fit_region_start = calibration.convert_from_calibrated_value(fit_interval_eV[0]) / dimensional_shape[-1]
                fit_region_end = calibration.convert_from_calibrated_value(fit_interval_eV[1]) / dimensional_shape[-1]
                signal_region_start = calibration.convert_from_calibrated_value(signal_interval_eV[0]) / dimensional_shape[-1]
                signal_region_end = calibration.convert_from_calibrated_value(signal_interval_eV[1]) / dimensional_shape[-1]
                fit_interval = fit_region_start, fit_region_end
                signal_interval = signal_region_start, signal_region_end
                return ElementalMappingController.ElementalMappingEdge(electron_shell=electron_shell, fit_interval=fit_interval, signal_interval=signal_interval)
        return None

    def __run_until_complete(self, document_controller):
        # run for 0.2s; recomputing and running periodic
        for _ in range(10):
            document_controller.document_model.recompute_all()
            document_controller.periodic()
            time.sleep(1/50)

    def test_explore_creates_initial_line_plot(self):
        with TestContext.create_memory_context() as test_context:
            document_controller = test_context.create_document_controller()
            document_model = document_controller.document_model
            elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
            data_item = self.__create_spectrum_image()
            document_model.append_data_item(data_item)
            elemental_mapping_controller.set_current_data_item(data_item)
            document_controller.event_loop.create_task(elemental_mapping_controller.explore_edges(document_controller))
            document_controller.periodic()  # start tasks
            document_model.recompute_all()
            document_controller.periodic()  # finish tasks
            self.assertEqual(2, len(document_model.data_items))
            explorer_data_item = document_model.data_items[1]
            explorer_display_item = document_model.get_display_item_for_data_item(explorer_data_item)
            self.assertEqual(1, len(explorer_data_item.dimensional_shape))
            self.assertEqual("explore", explorer_display_item.graphics[-1].graphic_id)

    def test_explore_adds_edge(self):
        with TestContext.create_memory_context() as test_context:
            document_controller = test_context.create_document_controller()
            document_model = document_controller.document_model
            elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
            model_data_item = self.__create_spectrum_image()
            document_model.append_data_item(model_data_item)
            elemental_mapping_controller.set_current_data_item(model_data_item)
            document_controller.event_loop.create_task(elemental_mapping_controller.explore_edges(document_controller))
            document_controller.periodic()  # start tasks
            document_model.recompute_all()
            document_controller.periodic()  # finish tasks
            explorer_data_item = document_model.data_items[1]
            explorer_display_item = document_model.get_display_item_for_data_item(explorer_data_item)
            elemental_mapping_controller.set_current_data_item(explorer_data_item)
            self.assertIsNotNone(elemental_mapping_controller.model_data_item)
            energy_calibration = explorer_data_item.dimensional_calibrations[-1]
            explorer_display_item.graphics[-1].interval = energy_calibration.convert_from_calibrated_value(1200) / 1024, energy_calibration.convert_from_calibrated_value(1226) / 1024
            for _ in range(3):
                # there is something funny about how async works; recent versions of Swift are faster
                # and have revealed some race condition about how items get added to the async queue.
                # to avoid that problem, do periodic over a period of a few ms.
                document_controller.periodic()  # update explorer interval
                time.sleep(0.01)
            edges = PeriodicTable.PeriodicTable().find_edges_in_energy_interval(elemental_mapping_controller.explorer_interval)
            elemental_mapping_controller.add_edge(edges[0])
            self.assertEqual(1, len(document_model.data_structures))
            edge_data_struct = document_model.data_structures[0]
            self.assertEqual("elemental_mapping_edge", edge_data_struct.structure_type)
            self.assertEqual(model_data_item, edge_data_struct.source)
            self.assertEqual(32, edge_data_struct.get_property_value("atomic_number"))
            self.assertEqual(2, edge_data_struct.get_property_value("shell_number"))
            self.assertEqual(3, edge_data_struct.get_property_value("subshell_index"))

    def test_adding_multiple_edges(self):
        with TestContext.create_memory_context() as test_context:
            document_controller = test_context.create_document_controller()
            document_model = document_controller.document_model
            elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
            model_data_item = self.__create_spectrum_image()
            document_model.append_data_item(model_data_item)
            elemental_mapping_controller.set_current_data_item(model_data_item)
            elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(32, 2, 3))  # Ge-L
            elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(14, 1, 1))  # Si-K
            self.assertEqual(2, len(document_model.data_structures))

    def test_removing_edges(self):
        with TestContext.create_memory_context() as test_context:
            document_controller = test_context.create_document_controller()
            document_model = document_controller.document_model
            elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
            model_data_item = self.__create_spectrum_image()
            document_model.append_data_item(model_data_item)
            elemental_mapping_controller.set_current_data_item(model_data_item)
            ge_edge = elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(32, 2, 3))  # Ge-L
            si_edge = elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(14, 1, 1))  # Si-K
            self.assertEqual(2, len(document_model.data_structures))
            elemental_mapping_controller.remove_edge(ge_edge)
            self.assertEqual(1, len(document_model.data_structures))
            self.assertEqual(14, document_model.data_structures[0].get_property_value("atomic_number"))
            elemental_mapping_controller.remove_edge(si_edge)
            self.assertEqual(0, len(document_model.data_structures))

    def test_controller_has_proper_edge_bundles_when_explorer_selected(self):
        with TestContext.create_memory_context() as test_context:
            document_controller = test_context.create_document_controller()
            document_model = document_controller.document_model
            elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
            model_data_item = self.__create_spectrum_image()
            document_model.append_data_item(model_data_item)
            elemental_mapping_controller.set_current_data_item(model_data_item)
            elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(14, 1, 1))  # Si-K
            document_controller.event_loop.create_task(elemental_mapping_controller.explore_edges(document_controller))
            self.__run_until_complete(document_controller)
            explorer_data_item = document_model.data_items[1]
            elemental_mapping_controller.set_current_data_item(explorer_data_item)
            self.assertIsNotNone(elemental_mapping_controller.model_data_item)
            edge_bundle = elemental_mapping_controller.build_edge_bundles(document_controller)
            self.assertEqual(1, len(edge_bundle))

    def test_picking_edge_produces_properly_configured_composite(self):
        with TestContext.create_memory_context() as test_context:
            document_controller = test_context.create_document_controller()
            document_model = document_controller.document_model
            elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
            model_data_item = self.__create_spectrum_image()
            document_model.append_data_item(model_data_item)
            model_display_item = document_model.get_display_item_for_data_item(model_data_item)
            elemental_mapping_controller.set_current_data_item(model_data_item)
            elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(14, 1, 1))  # Si-K
            edge_bundle = elemental_mapping_controller.build_edge_bundles(document_controller)
            edge_bundle[0].pick_action()
            self.__run_until_complete(document_controller)
            self.assertEqual(2, len(document_model.data_items))
            eels_data_item = document_model.data_items[1]
            self.assertEqual(model_display_item.graphics[0], eels_data_item.source)
            self.assertEqual(eels_data_item, document_model.data_structures[1].source)
            self.assertEqual("elemental_mapping_edge_ref", document_model.data_structures[1].structure_type)
            self.assertEqual(document_model.data_structures[0], document_model.data_structures[1].get_referenced_object("edge"))

    def test_deleting_pick_also_deletes_computation(self):
        with TestContext.create_memory_context() as test_context:
            document_controller = test_context.create_document_controller()
            document_model = document_controller.document_model
            elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
            model_data_item = self.__create_spectrum_image()
            document_model.append_data_item(model_data_item)
            elemental_mapping_controller.set_current_data_item(model_data_item)
            elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(14, 1, 1))  # Si-K
            edge_bundle = elemental_mapping_controller.build_edge_bundles(document_controller)
            edge_bundle[0].pick_action()
            self.__run_until_complete(document_controller)
            eels_data_item = document_model.data_items[1]
            self.assertEqual(1, len(document_model.computations))
            self.assertEqual(2, len(document_model.data_items))
            self.assertEqual(2, len(document_model.data_structures))
            document_model.remove_data_item(eels_data_item)
            self.assertEqual(0, len(document_model.computations))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(1, len(document_model.data_structures))

    def test_deleting_pick_region_also_deletes_pick_composition(self):
        with TestContext.create_memory_context() as test_context:
            document_controller = test_context.create_document_controller()
            document_model = document_controller.document_model
            elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
            model_data_item = self.__create_spectrum_image()
            document_model.append_data_item(model_data_item)
            model_display_item = document_model.get_display_item_for_data_item(model_data_item)
            elemental_mapping_controller.set_current_data_item(model_data_item)
            elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(14, 1, 1))  # Si-K
            edge_bundle = elemental_mapping_controller.build_edge_bundles(document_controller)
            edge_bundle[0].pick_action()
            self.__run_until_complete(document_controller)
            pick_region = model_display_item.graphics[0]
            eels_data_item = document_model.data_items[1]
            self.assertEqual(1, len(document_model.computations))
            self.assertEqual(2, len(document_model.data_items))
            self.assertEqual(2, len(document_model.data_structures))
            model_display_item.remove_graphic(pick_region)
            self.assertEqual(0, len(document_model.computations))
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(1, len(document_model.data_structures))

    def test_selecting_composite_updates_edge_value(self):
        with TestContext.create_memory_context() as test_context:
            document_controller = test_context.create_document_controller()
            document_model = document_controller.document_model
            elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
            model_data_item = self.__create_spectrum_image()
            document_model.append_data_item(model_data_item)
            elemental_mapping_controller.set_current_data_item(model_data_item)
            si_edge = elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(14, 1, 1))  # Si-K
            edge_bundle = elemental_mapping_controller.build_edge_bundles(document_controller)
            edge_bundle[0].pick_action()
            self.__run_until_complete(document_controller)
            self.assertIsNone(elemental_mapping_controller.edge)
            eels_data_item = document_model.data_items[1]
            elemental_mapping_controller.set_current_data_item(eels_data_item)
            self.assertEqual(model_data_item, elemental_mapping_controller.model_data_item)
            self.assertEqual(si_edge.data_structure, elemental_mapping_controller.edge.data_structure)

    def test_background_subtraction_computation_functions_reasonably(self):
        with TestContext.create_memory_context() as test_context:
            document_controller = test_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
            model_data_item = self.__create_spectrum_image()
            document_model.append_data_item(model_data_item)
            elemental_mapping_controller.set_current_data_item(model_data_item)
            si_edge = elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(14, 1, 1))  # Si-K
            spectrum_data_item = self.__create_spectrum()
            document_model.append_data_item(spectrum_data_item)
            spectrum_display_item = document_model.get_display_item_for_data_item(spectrum_data_item)
            spectrum_display_data_channel = spectrum_display_item.get_display_data_channel_for_data_item(spectrum_data_item)
            computation = document_model.create_computation()
            computation.create_input_item("eels_spectrum_xdata", Symbolic.make_item(spectrum_display_data_channel, type="display_xdata"))
            computation.create_input_item("fit_interval", Symbolic.make_item(si_edge.data_structure), property_name="fit_interval")
            computation.create_input_item("signal_interval", Symbolic.make_item(si_edge.data_structure), property_name="signal_interval")
            computation.processing_id = "eels.background_subtraction"
            document_model.append_computation(computation)
            document_model.recompute_all()
            document_controller.periodic()
            self.assertEqual(2, len(document_model.data_items))

    def test_changing_edge_configures_other_items_correctly(self):
        with TestContext.create_memory_context() as test_context:
            document_controller = test_context.create_document_controller()
            document_model = document_controller.document_model
            elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
            model_data_item = self.__create_spectrum_image()
            document_model.append_data_item(model_data_item)
            model_display_item = document_model.get_display_item_for_data_item(model_data_item)
            elemental_mapping_controller.set_current_data_item(model_data_item)
            elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(14, 1, 1))  # Si-K
            elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(32, 2, 3))  # Ge-L
            edge_bundle = elemental_mapping_controller.build_edge_bundles(document_controller)
            edge_bundle[0].pick_action()
            self.__run_until_complete(document_controller)
            eels_data_item = document_model.data_items[1]
            eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
            elemental_mapping_controller.set_current_data_item(eels_data_item)
            edge_bundle = elemental_mapping_controller.build_edge_bundles(document_controller)
            # apply the change to the other edge
            edge_bundle[1].select_action()
            self.__run_until_complete(document_controller)
            computation = document_model.computations[0]
            old_edge_data_structure = document_model.data_structures[0]
            new_edge_data_structure = document_model.data_structures[1]
            edge_ref_data_structure = document_model.data_structures[2]
            pick_region = model_display_item.graphics[0]
            # check the titles
            self.assertEqual("Pick Ge-L3", pick_region.label)
            self.assertEqual("Pick Ge-L3 EELS Data of Untitled", eels_data_item.title)
            # check the old intervals are disconnected and the new are connected
            old_fit_interval = eels_display_item.graphics[0].interval
            old_signal_interval = eels_display_item.graphics[1].interval
            new_fit_interval = (0.6, 0.7)
            new_signal_interval = (0.7, 0.8)
            # ensure changing old edge doesn't affect any connections
            old_edge_data_structure.set_property_value("fit_interval", new_fit_interval)
            old_edge_data_structure.set_property_value("signal_interval", new_signal_interval)
            self.assertEqual(old_fit_interval, eels_display_item.graphics[0].interval)
            self.assertEqual(old_signal_interval, eels_display_item.graphics[1].interval)
            self.assertEqual(old_fit_interval, computation.get_input("fit_interval"))
            self.assertEqual(old_signal_interval, computation.get_input("signal_interval"))
            # ensure changing new edge affects all connections
            new_edge_data_structure.set_property_value("fit_interval", new_fit_interval)
            new_edge_data_structure.set_property_value("signal_interval", new_signal_interval)
            self.assertEqual(new_fit_interval, eels_display_item.graphics[0].interval)
            self.assertEqual(new_signal_interval, eels_display_item.graphics[1].interval)
            self.assertEqual(new_fit_interval, computation.get_input("fit_interval"))
            self.assertEqual(new_signal_interval, computation.get_input("signal_interval"))
            # and the edge reference
            self.assertEqual(new_edge_data_structure, edge_ref_data_structure.get_referenced_object("edge"))

    def test_mapping_edge_produces_properly_configured_map(self):
        with TestContext.create_memory_context() as test_context:
            document_controller = test_context.create_document_controller()
            document_model = document_controller.document_model
            elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
            model_data_item = self.__create_spectrum_image()
            document_model.append_data_item(model_data_item)
            elemental_mapping_controller.set_current_data_item(model_data_item)
            elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(14, 1, 1))  # Si-K
            edge_bundle = elemental_mapping_controller.build_edge_bundles(document_controller)
            edge_bundle[0].map_action()
            self.__run_until_complete(document_controller)
            self.assertEqual(2, len(document_model.data_items))
            mapped_data_item = document_model.data_items[1]
            self.assertEqual(model_data_item, mapped_data_item.source)
            self.assertEqual(1, len(document_model.computations))
            self.assertEqual("eels.mapping", document_model.computations[0].processing_id)
            self.assertEqual(mapped_data_item.dimensional_calibrations, model_data_item.dimensional_calibrations[0:2])

    def test_multiprofile_of_two_maps_builds_two_line_profiles(self):
        with TestContext.create_memory_context() as test_context:
            document_controller = test_context.create_document_controller()
            document_model = document_controller.document_model
            elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
            model_data_item = self.__create_spectrum_image()
            document_model.append_data_item(model_data_item)
            elemental_mapping_controller.set_current_data_item(model_data_item)
            elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(14, 1, 1))  # Si-K
            elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(32, 2, 3))  # Ge-L
            edge_bundle = elemental_mapping_controller.build_edge_bundles(document_controller)
            edge_bundle[0].map_action()
            self.__run_until_complete(document_controller)
            edge_bundle[1].map_action()
            self.__run_until_complete(document_controller)
            self.assertEqual(3, len(document_model.data_items))
            elemental_mapping_controller.build_multiprofile(document_controller)
            self.assertEqual(5, len(document_model.data_items))
            self.assertEqual(6, len(document_model.display_items))
            composite_display_item = document_model.display_items[3]
            line_profile1_data_item = document_model.data_items[3]
            line_profile2_data_item = document_model.data_items[4]
            self.assertIn(line_profile1_data_item, composite_display_item.data_items)
            self.assertIn(line_profile2_data_item, composite_display_item.data_items)

    def test_multiprofile_of_two_maps_connects_line_profiles(self):
        with TestContext.create_memory_context() as test_context:
            document_controller = test_context.create_document_controller()
            document_model = document_controller.document_model
            elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
            model_data_item = self.__create_spectrum_image()
            document_model.append_data_item(model_data_item)
            elemental_mapping_controller.set_current_data_item(model_data_item)
            elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(14, 1, 1))  # Si-K
            elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(32, 2, 3))  # Ge-L
            edge_bundle = elemental_mapping_controller.build_edge_bundles(document_controller)
            edge_bundle[0].map_action()
            self.__run_until_complete(document_controller)
            edge_bundle[1].map_action()
            self.__run_until_complete(document_controller)
            elemental_mapping_controller.build_multiprofile(document_controller)
            map1_display_item = document_model.get_display_item_for_data_item(document_model.data_items[1])
            map2_display_item = document_model.get_display_item_for_data_item(document_model.data_items[2])
            # composite_data_item = document_model.data_items[3]
            # line_profile1_data_item = document_model.data_items[4]
            # line_profile2_data_item = document_model.data_items[5]
            line_region1 = map1_display_item.graphics[0]
            line_region2 = map2_display_item.graphics[0]
            self.assertEqual(line_region1.vector, line_region2.vector)
            self.assertEqual(line_region1.width, line_region2.width)
            line_region1.vector = (0.11, 0.12), (0.21, 0.22)
            self.assertEqual(line_region1.vector, line_region2.vector)
            self.assertEqual(line_region1.width, line_region2.width)

    def test_map_background_subtracted_signal_keeps_input_dtype(self):
        si_xdata_32 = self.__create_spectrum_image_xdata(dtype=numpy.float32)
        si_xdata_64 = self.__create_spectrum_image_xdata(dtype=numpy.float64)
        mapped_xdata_32 = eels_analysis.map_background_subtracted_signal(si_xdata_32, None, [(0.2, 0.3)], (0.4, 0.5))
        mapped_xdata_64 = eels_analysis.map_background_subtracted_signal(si_xdata_64, None, [(0.2, 0.3)], (0.4, 0.5))
        self.assertEqual(numpy.float32, mapped_xdata_32.data.dtype)
        self.assertEqual(numpy.float64, mapped_xdata_64.data.dtype)

    def test_align_zlp_keeps_input_dtype(self):
        si_xdata_32 = self.__create_spectrum_image_xdata(dtype=numpy.float32)
        si_xdata_64 = self.__create_spectrum_image_xdata(dtype=numpy.float64)
        mapped_xdata_32 = AlignZLP.align_zlp_xdata(si_xdata_32)[0]
        mapped_xdata_64 = AlignZLP.align_zlp_xdata(si_xdata_64)[0]
        self.assertEqual(numpy.float32, mapped_xdata_32.data.dtype)
        self.assertEqual(numpy.float64, mapped_xdata_64.data.dtype)

    def test_map_thickness_is_always_float32(self):
        si_xdata_32 = self.__create_spectrum_image_xdata(dtype=numpy.float32)
        si_xdata_64 = self.__create_spectrum_image_xdata(dtype=numpy.float64)
        mapped_xdata_32 = ThicknessMap.map_thickness_xdata(si_xdata_32)
        mapped_xdata_64 = ThicknessMap.map_thickness_xdata(si_xdata_64)
        self.assertEqual(numpy.float32, mapped_xdata_32.data.dtype)
        self.assertEqual(numpy.float32, mapped_xdata_64.data.dtype)
