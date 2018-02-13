# standard libraries
import collections
import copy
import functools
import gettext

# third party libraries
import numpy

# local libraries
from nion.swift.model import Connection
from nion.swift.model import DataItem
from nion.swift.model import DocumentModel
from nion.swift.model import Graphics
from nion.eels_analysis import PeriodicTable

_ = gettext.gettext


processing_descriptions = {
    "eels.eels_extract_signal":
        { 'script': 'from nion.eels_analysis import eels_analysis as ea\nfrom nion.data import xdata_1_0 as xd\ntarget.xdata = xd.vstack((ea.extract_signal_from_polynomial_background({src}, signal.interval, (fit.interval, )), {src}))',
          'sources': [{'label': 'Source', 'name': 'src', 'regions': [
              {'name': 'fit', 'params': {'label': 'Fit', 'interval': (0.2, 0.3)}, 'type': 'interval'},
              {'name': 'signal', 'params': {'label': 'Signal', 'interval': (0.4, 0.5)}, 'type': 'interval'},
          ]}],
          'title': 'Background Subtracted',
        },
    "eels.subtract_linear_background":
        { 'script': 'from nion.eels_analysis import eels_analysis as ea\nfrom nion.data import xdata_1_0 as xd\ntarget.xdata = xd.vstack((ea.subtract_linear_background({src}, fit.interval, (0, 1)), {src}))',
          'sources': [{'label': 'Source', 'name': 'src', 'regions': [
              {'name': 'fit', 'params': {'label': 'Fit', 'interval': (0.2, 0.3)}, 'type': 'interval'},
          ]}],
          'title': 'Linear Background Subtracted',
        },
    "eels.subtract_background_signal":
        { 'script': 'from nion.eels_analysis import eels_analysis as ea\nfrom nion.data import xdata_1_0 as xd\nsignal_xdata = ea.extract_original_signal({src}, fit.interval, signal.interval)\nbackground = ea.calculate_background_signal({src}, fit.interval, signal.interval)\ntarget.xdata = xd.vstack((signal_xdata, background, signal_xdata - background))',
          'sources': [{'label': 'Source', 'name': 'src', 'regions': [
              {'name': 'fit', 'params': {'label': 'Fit', 'interval': (0.2, 0.3)}, 'type': 'interval'},
              {'name': 'signal', 'params': {'label': 'Signal', 'interval': (0.4, 0.5)}, 'type': 'interval'},
          ]}],
          'title': 'Background Subtracted',
        },
    "eels.explore":
        { 'expression': 'xd.sum_region({src}, region.mask_xdata_with_shape({src}.data_shape[0:2]))',
          'sources': [
              {'name': 'src', 'label': 'Source', 'use_display_data': False, 'regions': [{'name': 'region', 'type': 'rectangle', 'params': {'label': 'Pick Region'}}], 'requirements': [{'type': 'dimensionality', 'min': 3, 'max': 3}]}
          ],
          'title': 'Explore',
          'out_regions': [
              {'name': 'interval_region', 'type': 'interval', 'params': {'label': 'Display Slice'}},
              {'name': 'explore', 'type': 'interval', 'params': {'label': 'Explore', 'interval': (0.4, 0.6), 'graphic_id': 'explore'}},
          ],
          'connections': [{'type': 'property', 'src': 'display', 'src_prop': 'slice_interval', 'dst': 'interval_region', 'dst_prop': 'interval'}]},
    }


def processing_extract_signal(document_controller):
    display_specifier = document_controller.selected_display_specifier
    data_item = document_controller.document_model.make_data_item_with_computation("eels.eels_extract_signal", [(display_specifier.data_item, None)], {"src": [None, None]})
    if data_item:
        new_display_specifier = DataItem.DisplaySpecifier.from_data_item(data_item)
        document_controller.display_data_item(new_display_specifier)
        return data_item
    return None


def processing_subtract_linear_background(document_controller):
    display_specifier = document_controller.selected_display_specifier
    data_item = document_controller.document_model.make_data_item_with_computation("eels.subtract_linear_background", [(display_specifier.data_item, None)], {"src": [None]})
    if data_item:
        new_display_specifier = DataItem.DisplaySpecifier.from_data_item(data_item)
        document_controller.display_data_item(new_display_specifier)
        return data_item
    return None


def processing_subtract_background_signal(document_controller):
    display_specifier = document_controller.selected_display_specifier
    data_item = document_controller.document_model.make_data_item_with_computation("eels.subtract_background_signal", [(display_specifier.data_item, None)], {"src": [None, None]})
    if data_item:
        new_display_specifier = DataItem.DisplaySpecifier.from_data_item(data_item)
        document_controller.display_data_item(new_display_specifier)
        return data_item
    return None


async def pick_new_edge(document_controller, model_data_item, edge) -> None:
    document_model = document_controller.document_model
    pick_region = Graphics.RectangleGraphic()
    pick_region.size = 16 / model_data_item.dimensional_shape[0], 16 / model_data_item.dimensional_shape[1]
    pick_region.label = "{} {}".format(_("Pick"), str(edge.electron_shell))
    model_data_item.displays[0].add_graphic(pick_region)

    pick_data_item = document_model.get_pick_region_new(model_data_item, pick_region=pick_region)
    if pick_data_item:
        pick_data_item.title = "{} Data of {}".format(pick_region.label, model_data_item.title)
        pick_display_specifier = DataItem.DisplaySpecifier.from_data_item(pick_data_item)
        pick_display_specifier.display.display_type = "line_plot"
        fit_region = Graphics.IntervalGraphic()
        fit_region.label = _("Fit")
        fit_region.graphic_id = "fit"
        fit_region.interval = edge.fit_interval
        pick_display_specifier.display.add_graphic(fit_region)
        signal_region = Graphics.IntervalGraphic()
        signal_region.label = _("Signal")
        signal_region.graphic_id = "signal"
        signal_region.interval = edge.signal_interval
        pick_display_specifier.display.add_graphic(signal_region)
        pick_data_item.add_connection(Connection.PropertyConnection(edge.data_structure, "fit_interval", fit_region, "interval"))
        pick_data_item.add_connection(Connection.PropertyConnection(edge.data_structure, "signal_interval", signal_region, "interval"))
        await document_model.recompute_immediate(document_controller.event_loop, pick_data_item)  # need the data to scale display; so do this here. ugh.

        background_data_item = DataItem.DataItem(numpy.zeros(1, ))
        background_data_item.title = "{} Background of {}".format(pick_region.label, model_data_item.title)
        background_display_specifier = DataItem.DisplaySpecifier.from_data_item(background_data_item)
        background_display_specifier.display.display_type = "line_plot"
        background_script = "from nion.eels_analysis import eels_analysis as ea\ntarget.xdata = ea.calculate_background_signal(pick.xdata, mapping.get_property('fit_interval'), mapping.get_property('signal_interval'))"
        background_computation = document_model.create_computation(background_script)
        background_computation.create_object("mapping", document_model.get_object_specifier(edge.data_structure), label="Mapping")
        background_computation.create_object("pick", document_model.get_object_specifier(pick_data_item))
        document_model.append_data_item(background_data_item)
        document_model.set_data_item_computation(background_data_item, background_computation)
        await document_model.recompute_immediate(document_controller.event_loop, background_data_item)  # need the data to scale display; so do this here. ugh.

        subtracted_data_item = DataItem.DataItem(numpy.zeros(1, ))
        subtracted_data_item.title = "{} Subtracted of {}".format(pick_region.label, model_data_item.title)
        subtracted_display_specifier = DataItem.DisplaySpecifier.from_data_item(subtracted_data_item)
        subtracted_display_specifier.display.display_type = "line_plot"
        subtracted_script = "from nion.eels_analysis import eels_analysis as ea\nsignal = ea.extract_original_signal(pick.xdata, mapping.get_property('fit_interval'), mapping.get_property('signal_interval'))\nbackground = ea.calculate_background_signal(pick.xdata, mapping.get_property('fit_interval'), mapping.get_property('signal_interval'))\ntarget.xdata = signal - background"
        subtracted_computation = document_model.create_computation(subtracted_script)
        subtracted_computation.create_object("mapping", document_model.get_object_specifier(edge.data_structure), label="Mapping")
        subtracted_computation.create_object("pick", document_model.get_object_specifier(pick_data_item))
        document_model.append_data_item(subtracted_data_item)
        document_model.set_data_item_computation(subtracted_data_item, subtracted_computation)
        await document_model.recompute_immediate(document_controller.event_loop, subtracted_data_item)  # need the data to scale display; so do this here. ugh.

        composite_data_item = DataItem.CompositeLibraryItem()
        composite_data_item.title = "{} from {}".format(pick_region.label, model_data_item.title)
        composite_data_item.append_data_item(pick_data_item)
        composite_data_item.append_data_item(background_data_item)
        composite_data_item.append_data_item(subtracted_data_item)
        pick_data_item.source = composite_data_item
        subtracted_data_item.source = composite_data_item
        pick_data_item.source = composite_data_item
        composite_display_specifier = DataItem.DisplaySpecifier.from_data_item(composite_data_item)
        composite_display_specifier.display.display_type = "line_plot"
        composite_display_specifier.display.dimensional_scales = (model_data_item.dimensional_shape[-1], )
        composite_display_specifier.display.dimensional_calibrations = (model_data_item.dimensional_calibrations[-1], )
        composite_display_specifier.display.intensity_calibration = model_data_item.intensity_calibration
        document_model.append_data_item(composite_data_item)
        fit_region = Graphics.IntervalGraphic()
        fit_region.label = _("Fit")
        fit_region.graphic_id = "fit"
        fit_region.interval = edge.fit_interval
        composite_display_specifier.display.add_graphic(fit_region)
        signal_region = Graphics.IntervalGraphic()
        signal_region.label = _("Signal")
        signal_region.graphic_id = "signal"
        signal_region.interval = edge.signal_interval
        composite_display_specifier.display.add_graphic(signal_region)
        composite_data_item.add_connection(Connection.PropertyConnection(edge.data_structure, "fit_interval", fit_region, "interval"))
        composite_data_item.add_connection(Connection.PropertyConnection(edge.data_structure, "signal_interval", signal_region, "interval"))
        composite_display_specifier.display.view_to_intervals(pick_data_item.xdata, [edge.fit_interval, edge.signal_interval])
        document_controller.display_data_item(composite_display_specifier)

    return pick_data_item


def map_new_edge(document_controller, model_data_item, edge):
    document_model = document_controller.document_model

    map_data_item = DataItem.new_data_item()
    map_data_item.title = "{} of {}".format(_("Map"), str(edge.electron_shell))
    map_data_item.category = model_data_item.category
    document_model.append_data_item(map_data_item)
    display_specifier = DataItem.DisplaySpecifier.from_data_item(map_data_item)
    script = """from nion.eels_analysis import eels_analysis as ea
from nion.eels_analysis import PeriodicTable as pt
electron_shell = pt.ElectronShell(mapping.get_property('atomic_number'), mapping.get_property('shell_number'), mapping.get_property('subshell_index'))
target.xdata = ea.map_background_subtracted_signal(src.xdata, electron_shell, mapping.get_property('fit_interval'), mapping.get_property('signal_interval'))
"""
    computation = document_model.create_computation(script)
    computation.label = "EELS Map"
    computation.create_object("src", document_model.get_object_specifier(model_data_item), label="Source")
    computation.create_object("mapping", document_model.get_object_specifier(edge.data_structure), label="Mapping")
    document_model.set_data_item_computation(display_specifier.data_item, computation)

    document_controller.display_data_item(DataItem.DisplaySpecifier.from_data_item(map_data_item))

    return map_data_item


class ElementalMappingEdge:
    def __init__(self, *, data_structure: DocumentModel.DataStructure=None, electron_shell: PeriodicTable.ElectronShell=None, fit_interval=None, signal_interval=None):
        self.__data_structure = data_structure
        self.__fit_interval = fit_interval
        self.__signal_interval = signal_interval
        self.__electron_shell = electron_shell
        if self.__data_structure:
            self.read(self.__data_structure)

    @property
    def data_structure(self) -> DocumentModel.DataStructure:
        return self.__data_structure

    def read(self, data_structure: DocumentModel.DataStructure) -> None:
        atomic_number = data_structure.get_property_value("atomic_number")
        shell_number = data_structure.get_property_value("shell_number")
        subshell_index = data_structure.get_property_value("subshell_index")
        self.__electron_shell = PeriodicTable.ElectronShell(atomic_number, shell_number, subshell_index)
        self.__fit_interval = data_structure.get_property_value("fit_interval", (0.4, 0.5))
        self.__signal_interval = data_structure.get_property_value("signal_interval", (0.5, 0.6))

    def write(self, data_structure: DocumentModel.DataStructure) -> None:
        self.__write_electron_shell(data_structure)
        self.__write_fit_interval(data_structure)
        self.__write_signal_interval(data_structure)

    def matches(self, data_structure: DocumentModel.DataStructure) -> bool:
        return self.__data_structure is not None and self.__data_structure.uuid == data_structure.uuid

    def __write_electron_shell(self, data_structure):
        if self.__electron_shell:
            data_structure.set_property_value("atomic_number", self.__electron_shell.atomic_number)
            data_structure.set_property_value("shell_number", self.__electron_shell.shell_number)
            data_structure.set_property_value("subshell_index", self.__electron_shell.subshell_index)
        else:
            data_structure.remove_property_value("atomic_number")
            data_structure.remove_property_value("shell_number")
            data_structure.remove_property_value("subshell_index")

    def __write_signal_interval(self, data_structure):
        if self.__signal_interval is not None:
            data_structure.set_property_value("signal_interval", copy.copy(self.__signal_interval))
        else:
            data_structure.remove_property_value("signal_interval")

    def __write_fit_interval(self, data_structure):
        if self.__fit_interval is not None:
            data_structure.set_property_value("fit_interval", copy.copy(self.__fit_interval))
        else:
            data_structure.remove_property_value("fit_interval")

    @property
    def electron_shell(self):
        return self.__electron_shell

    @electron_shell.setter
    def electron_shell(self, value):
        if self.__electron_shell != value:
            self.__electron_shell = value
            self.__write_electron_shell(self.__data_structure)

    @property
    def fit_interval(self):
        return self.__fit_interval

    @fit_interval.setter
    def fit_interval(self, value):
        if self.__fit_interval != value:
            self.__fit_interval = value
            self.__write_fit_interval(self.__data_structure)

    @property
    def signal_interval(self):
        return self.__signal_interval

    @signal_interval.setter
    def signal_interval(self, value):
        if self.__signal_interval != value:
            self.__signal_interval = value
            self.__write_signal_interval(self.__data_structure)


class ElementalMappingController:
    def __init__(self, document_model: DocumentModel.DocumentModel):
        self.__document_model = document_model

        self.__current_data_item = None
        self.__model_data_item = None
        self.__edge_data_structure = None

        self.__explorer_interval = None

        self.__explorer_property_changed_listeners = dict()  # typing.Dict[uuid.UUID, Any]

        self.__energy_intervals = dict()  # typing.Dict[uuid.UUID, typing.Tuple[float, float]]

        def item_inserted(key, value, before_index):
            if key == "data_item":
                data_item = value
                if self.__is_explorer(data_item):
                    self.connect_explorer_interval(data_item)

        def item_removed(key, value, index):
            if key == "data_item":
                data_item = value
                self.disconnect_explorer_interval(data_item)

        self.__item_inserted_listener = document_model.item_inserted_event.listen(item_inserted)
        self.__item_removed_listener = document_model.item_removed_event.listen(item_removed)

        for index, data_item in enumerate(document_model.data_items):
            item_inserted("data_item", data_item, index)

    def close(self):
        self.__item_inserted_listener.close()
        self.__item_inserted_listener = None
        self.__item_removed_listener.close()
        self.__item_removed_listener = None

    def set_current_data_item(self, data_item):
        """Set the current data item.

        If the data item is an explorer, update the explorer interval, otherwise cleaar it.
        """
        self.__current_data_item = data_item

        if self.__is_explorer(data_item):
            self.__explorer_interval = self.__energy_intervals.get(data_item.uuid)
        else:
            self.__explorer_interval = None

        self.__model_data_item = None
        self.__edge_data_structure = None

        if self.__is_model(data_item):
            self.__model_data_item = data_item
        else:
            computation = data_item.computation if data_item else None
            if computation:
                for computation_variable in computation.variables:
                    if computation_variable.name == "src":
                        src_data_item_value = self.__document_model.resolve_object_specifier(computation_variable.specifier)
                        src_data_item = src_data_item_value.value.data_item if src_data_item_value else None
                        if self.__is_model(src_data_item):
                            self.__model_data_item = src_data_item
                    if computation_variable.name == "mapping":
                        current_edge_value = self.__document_model.resolve_object_specifier(computation_variable.specifier)
                        self.__edge_data_structure = current_edge_value.value if current_edge_value else None

    @property
    def model_data_item(self):
        return self.__model_data_item

    def __explorer_interval_changed(self, data_item, interval) -> None:
        if data_item == self.__current_data_item:
            self.__explorer_interval = interval

    @property
    def explorer_interval(self):
        return self.__explorer_interval

    def __is_model(self, data_item) -> bool:
        if isinstance(data_item, DataItem.DataItem):
            return data_item.is_data_3d
        return False

    def __is_explorer(self, data_item) -> bool:
        if isinstance(data_item, DataItem.DataItem):
            if data_item.is_data_1d:
                for graphic in data_item.displays[0].graphics:
                    if isinstance(graphic, Graphics.IntervalGraphic) and graphic.graphic_id == "explore":
                        return True
        return False

    def __is_calibrated_map(self, data_item) -> bool:
        if isinstance(data_item, DataItem.DataItem):
            if data_item.is_data_2d:
                return data_item.title.startswith("Map") and data_item.intensity_calibration.units.startswith("~")
        return False

    async def explore_edges(self, document_controller):
        model_data_item = self.__model_data_item
        pick_region = Graphics.RectangleGraphic()
        pick_region.size = 16 / model_data_item.dimensional_shape[0], 16 / model_data_item.dimensional_shape[1]
        pick_region.label = _("Explore")
        model_data_item.displays[0].add_graphic(pick_region)
        pick_data_item = self.__document_model.make_data_item_with_computation("eels.explore", [(model_data_item, None)], {"src": [pick_region]})
        if pick_data_item:
            new_display_specifier = DataItem.DisplaySpecifier.from_data_item(pick_data_item)
            document_controller.display_data_item(new_display_specifier)
            await self.__document_model.recompute_immediate(document_controller.event_loop, pick_data_item)  # need the data to make connect_explorer_interval work; so do this here. ugh.
            self.connect_explorer_interval(pick_data_item)

    def __get_edges(self, data_item):
        edges = list()
        for data_structure in copy.copy(self.__document_model.data_structures):
            if data_structure.source == data_item and data_structure.structure_type == "elemental_mapping_edge":
                edge = ElementalMappingEdge(data_structure=data_structure)
                edges.append(edge)
        return edges

    def __add_edge(self, data_item, electron_shell, fit_interval, signal_interval):
        data_structure = self.__document_model.create_data_structure(structure_type="elemental_mapping_edge", source=data_item)
        self.__document_model.append_data_structure(data_structure)
        edge = ElementalMappingEdge(electron_shell=electron_shell, fit_interval=fit_interval, signal_interval=signal_interval)
        edge.write(data_structure)

    def __remove_edge(self, data_item, edge: ElementalMappingEdge) -> None:
        for data_structure in copy.copy(self.__document_model.data_structures):
            if data_structure.source == data_item and edge.matches(data_structure):
                self.__document_model.remove_data_structure(data_structure)

    def graphic_property_changed(self, graphic, data_item, dimensional_shape, dimensional_calibrations, key):
        if key == "interval":
            value = graphic.interval
            ss = value[0] * dimensional_shape[-1]
            ee = value[1] * dimensional_shape[-1]
            s = dimensional_calibrations[-1].convert_to_calibrated_value(ss)
            e = dimensional_calibrations[-1].convert_to_calibrated_value(ee)
            self.__energy_intervals[data_item.uuid] = s, e
            self.__explorer_interval_changed(data_item, (s, e))

    def connect_explorer_interval(self, data_item):
        if data_item.is_data_1d:
            for graphic in data_item.displays[0].graphics:
                if isinstance(graphic, Graphics.IntervalGraphic) and graphic.graphic_id == "explore":
                    dimensional_shape = data_item.dimensional_shape
                    dimensional_calibrations = data_item.dimensional_calibrations
                    self.__explorer_property_changed_listeners[data_item.uuid] = graphic.property_changed_event.listen(functools.partial(self.graphic_property_changed, graphic, data_item, dimensional_shape, dimensional_calibrations))
                    self.graphic_property_changed(graphic, data_item, dimensional_shape, dimensional_calibrations, "interval")

    def disconnect_explorer_interval(self, data_item):
        listener = self.__explorer_property_changed_listeners.get(data_item.uuid)
        if listener:
            listener.close()
            del self.__explorer_property_changed_listeners[data_item.uuid]

    def add_edge(self, model_data_item: DataItem.DataItem, electron_shell: PeriodicTable.ElectronShell, data_item: DataItem.DataItem) -> None:
        binding_energy_eV = PeriodicTable.PeriodicTable().nominal_binding_energy_ev(electron_shell)
        signal_interval_eV = binding_energy_eV, binding_energy_eV * 1.10
        fit_interval_eV = binding_energy_eV * 0.93, binding_energy_eV * 0.98
        dimensional_shape = data_item.dimensional_shape
        dimensional_calibrations = data_item.dimensional_calibrations
        if dimensional_shape is not None and dimensional_calibrations is not None and len(dimensional_calibrations) > 0:
            calibration = dimensional_calibrations[-1]
            if calibration.units == "eV":
                fit_region_start = calibration.convert_from_calibrated_value(fit_interval_eV[0]) / dimensional_shape[-1]
                fit_region_end = calibration.convert_from_calibrated_value(fit_interval_eV[1]) / dimensional_shape[-1]
                signal_region_start = calibration.convert_from_calibrated_value(signal_interval_eV[0]) / dimensional_shape[-1]
                signal_region_end = calibration.convert_from_calibrated_value(signal_interval_eV[1]) / dimensional_shape[-1]
                fit_interval = fit_region_start, fit_region_end
                signal_interval = signal_region_start, signal_region_end
                self.__add_edge(model_data_item, electron_shell, fit_interval, signal_interval)

    def build_edge_bundles(self, document_controller):
        document_model = self.__document_model
        model_data_item = self.__model_data_item
        current_data_item = self.__current_data_item
        edge_data_structure = self.__edge_data_structure

        EdgeBundle = collections.namedtuple("EdgeBundle", ["electron_shell_str", "selected", "select_action", "pick_action", "map_action", "delete_action"])

        edge_bundles = list()

        for index, edge in enumerate(self.__get_edges(model_data_item)):

            def change_edge_action(edge):
                document_controller.event_loop.create_task(self.__change_edge(document_controller.event_loop, document_model, model_data_item, current_data_item, edge))

            def pick_edge_action(edge):
                document_controller.event_loop.create_task(pick_new_edge(document_controller, model_data_item, edge))

            def map_edge_action(edge):
                map_new_edge(document_controller, model_data_item, edge)

            def delete_edge_action(edge):
                self.__remove_edge(model_data_item, edge)

            edge_bundle = EdgeBundle(electron_shell_str=edge.electron_shell.to_long_str(),
                                     selected=edge.data_structure == edge_data_structure,
                                     select_action=functools.partial(change_edge_action, edge),
                                     pick_action=functools.partial(pick_edge_action, edge),
                                     map_action=functools.partial(map_edge_action, edge),
                                     delete_action=functools.partial(delete_edge_action, edge))

            edge_bundles.append(edge_bundle)

        return edge_bundles

    async def __change_edge(self, event_loop, document_model, model_data_item, data_item, edge):
        mapping_computation_variable = None
        pick_region_specifier = None
        computation = data_item.computation if data_item else None
        if computation:
            for computation_variable in computation.variables:
                if computation_variable.name == "mapping":
                    mapping_computation_variable = computation_variable
                if computation_variable.name == "region":
                    pick_region_specifier = computation_variable.specifier
        if mapping_computation_variable:
            mapping_computation_variable.specifier = document_model.get_object_specifier(edge.data_structure)
            for connection in copy.copy(data_item.connections):
                if connection.source_property in ("fit_interval", "signal_interval"):
                    source_property = connection.source_property
                    target_property = connection.target_property
                    target = connection._target
                    data_item.remove_connection(connection)
                    new_connection = Connection.PropertyConnection(edge.data_structure, source_property, target, target_property)
                    data_item.add_connection(new_connection)
            if pick_region_specifier:
                pick_region_value = document_model.resolve_object_specifier(pick_region_specifier)
                if pick_region_value:
                    pick_region = pick_region_value.value
                    pick_region.label = "{} {}".format(_("Pick"), str(edge.electron_shell))
                    data_item.title = "{} of {}".format(pick_region.label, model_data_item.title)
            else:
                    data_item.title = "{} {} of {}".format(_("Map"), str(edge.electron_shell), model_data_item.title)
            document_model.rebind_computations()
        display = data_item.displays[0]
        if display.display_type == "line_plot":
            intervals = list()
            for graphic in display.graphics:
                if isinstance(graphic, Graphics.IntervalGraphic) and graphic.graphic_id in ("fit", "signal"):
                    intervals.append(graphic.interval)
            await document_model.recompute_immediate(event_loop, data_item)  # need the data to scale display; so do this here. ugh.
            display.view_to_intervals(data_item.xdata, intervals)

    def build_multiprofile(self, document_controller):
        model_data_item = self.__model_data_item
        document_model = self.__document_model
        multiprofile_data_item = None
        multiprofile_computation = None
        indexes = list()
        legend_labels = list()
        line_profile_regions = list()
        for index, dependent_data_item in enumerate(document_model.get_dependent_data_items(model_data_item)):
            if self.__is_calibrated_map(dependent_data_item):
                if not multiprofile_data_item:
                    multiprofile_data_item = DataItem.new_data_item()
                    multiprofile_computation = document_model.create_computation("")
                    multiprofile_computation.label = "EELS Multiprofile"
                indexes.append(index)
                legend_labels.append(dependent_data_item.title[dependent_data_item.title.index(" of ") + 4:])
                display = dependent_data_item.displays[0]
                line_profile_region = Graphics.LineProfileGraphic()
                line_profile_region.start = 0.5, 0.2
                line_profile_region.end = 0.5, 0.8
                display.add_graphic(line_profile_region)
                line_profile_regions.append(line_profile_region)
                multiprofile_computation.create_object("src" + str(index), document_model.get_object_specifier(dependent_data_item), label="Src" + str(index))
                multiprofile_computation.create_object("region" + str(index), document_model.get_object_specifier(line_profile_region), label="Region" + str(index))
        if multiprofile_data_item:
            script = "from nion.eels_analysis import eels_analysis as ea\nfrom nion.data import xdata_1_0 as xd\nimport numpy\n"
            for index in indexes:
                script += "d{0} = xd.line_profile(src{0}.display_xdata, region{0}.vector, region{0}.line_width)\n".format(index)
            profiles = ",".join(["d{0}".format(index) for index in indexes])
            script += "mx=numpy.amax(xd.vstack([{}]).data)\n".format(profiles)
            for index in indexes:
                script += "d{0} /= mx\n".format(index)
            script += "target.xdata = xd.vstack([{}])".format(profiles)
            multiprofile_computation.expression = script
            multiprofile_display_specifier = DataItem.DisplaySpecifier.from_data_item(multiprofile_data_item)
            multiprofile_display_specifier.display.display_type = "line_plot"
            multiprofile_display_specifier.display.legend_labels = legend_labels
            document_model.append_data_item(multiprofile_data_item)
            document_model.set_data_item_computation(multiprofile_data_item, multiprofile_computation)
            for line_profile_region in line_profile_regions[1:]:
                multiprofile_data_item.add_connection(Connection.PropertyConnection(line_profile_regions[0], "vector", line_profile_region, "vector"))
                multiprofile_data_item.add_connection(Connection.PropertyConnection(line_profile_regions[0], "width", line_profile_region, "width"))
            multiprofile_data_item.title = _("Profiles of ") + ", ".join(legend_labels)
            document_controller.display_data_item(multiprofile_display_specifier)

DocumentModel.DocumentModel.register_processing_descriptions(processing_descriptions)