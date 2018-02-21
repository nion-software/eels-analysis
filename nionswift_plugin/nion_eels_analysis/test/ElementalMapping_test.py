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
from nion.swift import DocumentController
from nion.swift import Facade
from nion.swift.model import DataItem
from nion.swift.model import DocumentModel
from nion.ui import TestUI

from nion.eels_analysis import PeriodicTable
from nionswift_plugin.nion_eels_analysis import ElementalMappingController


Facade.initialize()


class TestElementalMappingController(unittest.TestCase):

    def setUp(self):
        self.app = Application.Application(TestUI.UserInterface(), set_global=False)

    def tearDown(self):
        pass

    def __create_spectrum_image(self) -> DataItem.DataItem:
        data = numpy.zeros((8, 8, 1024))
        for row in range(data.shape[0]):
            for column in range(data.shape[1]):
                data[row, column, :] = numpy.random.uniform(10, 1000, 1024)
        intensity_calibration = Calibration.Calibration(units="~")
        dimensional_calibrations = [Calibration.Calibration(units="nm"), Calibration.Calibration(units="nm"), Calibration.Calibration(scale=2.0, units="eV")]
        data_descriptor = DataAndMetadata.DataDescriptor(is_sequence=False, collection_dimension_count=2, datum_dimension_count=1)
        xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)
        return DataItem.new_data_item(xdata)

    def __create_spectrum(self) -> DataItem.DataItem:
        data = numpy.random.uniform(10, 1000, 1024)
        intensity_calibration = Calibration.Calibration(units="~")
        dimensional_calibrations = [Calibration.Calibration(scale=2.0, units="eV")]
        data_descriptor = DataAndMetadata.DataDescriptor(is_sequence=False, collection_dimension_count=0, datum_dimension_count=1)
        xdata = DataAndMetadata.new_data_and_metadata(data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, data_descriptor=data_descriptor)
        return DataItem.new_data_item(xdata)

    def __run_until_complete(self, document_controller):
        # run for 0.2s; recomputing and running periodic
        for _ in range(10):
            document_controller.document_model.recompute_all()
            document_controller.periodic()
            time.sleep(1/50)

    def test_explore_creates_initial_line_plot(self):
        document_model = DocumentModel.DocumentModel()
        elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
        document_controller = DocumentController.DocumentController(self.app.ui, document_model, workspace_id="library")
        with contextlib.closing(document_controller), contextlib.closing(elemental_mapping_controller):
            data_item = self.__create_spectrum_image()
            document_model.append_data_item(data_item)
            elemental_mapping_controller.set_current_data_item(data_item)
            document_controller.event_loop.create_task(elemental_mapping_controller.explore_edges(document_controller))
            document_controller.periodic()  # start tasks
            document_model.recompute_all()
            document_controller.periodic()  # finish tasks
            self.assertEqual(2, len(document_model.data_items))
            self.assertEqual(1, len(document_model.data_items[1].dimensional_shape))
            self.assertEqual("explore", document_model.data_items[1].displays[0].graphics[-1].graphic_id)

    def test_explore_adds_edge(self):
        document_model = DocumentModel.DocumentModel()
        elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
        document_controller = DocumentController.DocumentController(self.app.ui, document_model, workspace_id="library")
        with contextlib.closing(document_controller), contextlib.closing(elemental_mapping_controller):
            model_data_item = self.__create_spectrum_image()
            document_model.append_data_item(model_data_item)
            elemental_mapping_controller.set_current_data_item(model_data_item)
            document_controller.event_loop.create_task(elemental_mapping_controller.explore_edges(document_controller))
            document_controller.periodic()  # start tasks
            document_model.recompute_all()
            document_controller.periodic()  # finish tasks
            explorer_data_item = document_model.data_items[1]
            elemental_mapping_controller.set_current_data_item(explorer_data_item)
            self.assertIsNotNone(elemental_mapping_controller.model_data_item)
            energy_calibration = explorer_data_item.dimensional_calibrations[-1]
            explorer_data_item.displays[0].graphics[-1].interval = energy_calibration.convert_from_calibrated_value(1200) / 1024, energy_calibration.convert_from_calibrated_value(1226) / 1024
            document_controller.periodic()  # update explorer interval
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
        document_model = DocumentModel.DocumentModel()
        elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
        document_controller = DocumentController.DocumentController(self.app.ui, document_model, workspace_id="library")
        with contextlib.closing(document_controller), contextlib.closing(elemental_mapping_controller):
            model_data_item = self.__create_spectrum_image()
            document_model.append_data_item(model_data_item)
            elemental_mapping_controller.set_current_data_item(model_data_item)
            elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(32, 2, 3))  # Ge-L
            elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(14, 1, 1))  # Si-K
            self.assertEqual(2, len(document_model.data_structures))

    def test_removing_edges(self):
        document_model = DocumentModel.DocumentModel()
        elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
        document_controller = DocumentController.DocumentController(self.app.ui, document_model, workspace_id="library")
        with contextlib.closing(document_controller), contextlib.closing(elemental_mapping_controller):
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
        document_model = DocumentModel.DocumentModel()
        elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
        document_controller = DocumentController.DocumentController(self.app.ui, document_model, workspace_id="library")
        with contextlib.closing(document_controller), contextlib.closing(elemental_mapping_controller):
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
        document_model = DocumentModel.DocumentModel()
        elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
        document_controller = DocumentController.DocumentController(self.app.ui, document_model, workspace_id="library")
        with contextlib.closing(document_controller), contextlib.closing(elemental_mapping_controller):
            model_data_item = self.__create_spectrum_image()
            document_model.append_data_item(model_data_item)
            elemental_mapping_controller.set_current_data_item(model_data_item)
            elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(14, 1, 1))  # Si-K
            edge_bundle = elemental_mapping_controller.build_edge_bundles(document_controller)
            edge_bundle[0].pick_action()
            self.__run_until_complete(document_controller)
            self.assertEqual(5, len(document_model.data_items))
            data_item = document_model.data_items[1]
            background_data_item = document_model.data_items[2]
            subtracted_data_item = document_model.data_items[3]
            composite_data_item = document_model.data_items[4]
            self.assertEqual(model_data_item, composite_data_item.source)
            self.assertEqual(composite_data_item, data_item.source)
            self.assertEqual(composite_data_item, background_data_item.source)
            self.assertEqual(composite_data_item, subtracted_data_item.source)
            self.assertEqual(composite_data_item, document_model.data_structures[1].source)
            self.assertEqual("elemental_mapping_edge_ref", document_model.data_structures[1].structure_type)
            self.assertEqual(document_model.data_structures[0], document_model.data_structures[1].get_referenced_object("edge"))

    def test_selecting_composite_updates_edge_value(self):
        document_model = DocumentModel.DocumentModel()
        elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
        document_controller = DocumentController.DocumentController(self.app.ui, document_model, workspace_id="library")
        with contextlib.closing(document_controller), contextlib.closing(elemental_mapping_controller):
            model_data_item = self.__create_spectrum_image()
            document_model.append_data_item(model_data_item)
            elemental_mapping_controller.set_current_data_item(model_data_item)
            si_edge = elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(14, 1, 1))  # Si-K
            edge_bundle = elemental_mapping_controller.build_edge_bundles(document_controller)
            edge_bundle[0].pick_action()
            self.__run_until_complete(document_controller)
            self.assertIsNone(elemental_mapping_controller.edge)
            composite_data_item = document_model.data_items[4]
            elemental_mapping_controller.set_current_data_item(composite_data_item)
            self.assertEqual(model_data_item, elemental_mapping_controller.model_data_item)
            self.assertEqual(si_edge.data_structure, elemental_mapping_controller.edge.data_structure)

    def test_background_subtraction_computation_functions_reasonably(self):
        document_model = DocumentModel.DocumentModel()
        self.app._set_document_model(document_model)  # required to allow API to find document model
        elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)
        document_controller = DocumentController.DocumentController(self.app.ui, document_model, workspace_id="library")
        with contextlib.closing(document_controller), contextlib.closing(elemental_mapping_controller):
            model_data_item = self.__create_spectrum_image()
            document_model.append_data_item(model_data_item)
            elemental_mapping_controller.set_current_data_item(model_data_item)
            si_edge = elemental_mapping_controller.add_edge(PeriodicTable.ElectronShell(14, 1, 1))  # Si-K
            spectrum_data_item = self.__create_spectrum()
            document_model.append_data_item(spectrum_data_item)
            computation = document_model.create_computation()
            computation.create_object("eels_spectrum_xdata", document_model.get_object_specifier(spectrum_data_item, "display_xdata"))
            computation.create_input("fit_interval", document_model.get_object_specifier(si_edge.data_structure), "fit_interval")
            computation.create_input("signal_interval", document_model.get_object_specifier(si_edge.data_structure), "signal_interval")
            computation.processing_id = "eels.background_subtraction"
            document_model.append_computation(computation)
            document_model.recompute_all()
            document_controller.periodic()
            self.assertEqual(4, len(document_model.data_items))
