# standard libraries
import collections
import copy
import functools
import gettext
import typing

# third party libraries
import numpy

# local libraries
from nion.data import xdata_1_0 as xd
from nion.swift import DocumentController
from nion.swift.model import Connection
from nion.swift.model import DataItem
from nion.swift.model import DisplayItem
from nion.swift.model import DocumentModel
from nion.swift.model import Graphics
from nion.swift.model import Symbolic
from nion.eels_analysis import eels_analysis
from nion.eels_analysis import PeriodicTable

_ = gettext.gettext


class EELSBackgroundSubtraction:
    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, eels_xdata, region, fit_interval, signal_interval):
        eels_spectrum_xdata = xd.sum_region(eels_xdata, region.mask_xdata_with_shape(eels_xdata.data_shape[0:2]))
        signal = eels_analysis.make_signal_like(eels_analysis.extract_original_signal(eels_spectrum_xdata, [fit_interval], signal_interval), eels_spectrum_xdata)
        background_xdata = eels_analysis.make_signal_like(eels_analysis.calculate_background_signal(eels_spectrum_xdata, [fit_interval], signal_interval), eels_spectrum_xdata)
        subtracted_xdata = signal - background_xdata
        self.__xdata = xd.vstack((eels_spectrum_xdata, background_xdata, subtracted_xdata))

    def commit(self):
        self.computation.set_referenced_xdata("data", self.__xdata)


async def pick_new_edge(document_controller, model_data_item, edge) -> None:
    """Set up a new edge pick from the model data item and the given edge.

    The library will have the following new components and connections:
        - a pick region on the model data item
        - a pick data item with a fit/signal connected to the edge data structure
        - a background subtraction computation with model data item, and edge intervals as inputs
        - a background data item, computed by the background subtraction computation
        - a subtracted data item, computed by the background subtraction computation
        - a eels line plot with pick, background, and subtracted data items as components
        - an edge reference, owned by eels line plot, with reference to edge
        - the edge reference is used to recognize the eels line plot as associated with the referenced edge
    """
    document_model = document_controller.document_model
    project = document_model._project
    model_display_item = document_model.get_display_item_for_data_item(model_data_item)
    pick_region = Graphics.RectangleGraphic()
    pick_region.size = min(1 / 16, 16 / model_data_item.dimensional_shape[0]), min(1 / 16, 16 / model_data_item.dimensional_shape[1])
    pick_region.label = "{} {}".format(_("Pick"), str(edge.electron_shell))
    model_display_item.add_graphic(pick_region)

    # set up the computation for this edge.
    eels_data_item = DataItem.DataItem()
    document_model.append_data_item(eels_data_item)
    eels_data_item.title = "{} EELS Data of {}".format(pick_region.label, model_data_item.title)
    eels_data_item.source = pick_region
    eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
    eels_display_item.display_type = "line_plot"
    eels_display_item.display_layers = [
        {"label": "Signal", "data_index": 0, "data_row": 2, "fill_color": "#0F0"},
        {"label": "Background", "data_index": 0, "data_row": 1, "fill_color": "rgba(255, 0, 0, 0.3)"},
        {"label": "Data", "data_index": 0, "data_row": 0, "fill_color": "#1E90FF"},
    ]
    eels_display_item.set_display_property("legend_position", "top-right")
    fit_region = Graphics.IntervalGraphic()
    fit_region.label = _("Fit")
    fit_region.graphic_id = "fit"
    fit_region.interval = edge.fit_interval
    eels_display_item.add_graphic(fit_region)
    signal_region = Graphics.IntervalGraphic()
    signal_region.label = _("Signal")
    signal_region.graphic_id = "signal"
    signal_region.interval = edge.signal_interval
    eels_display_item.add_graphic(signal_region)
    document_model.append_connection(Connection.PropertyConnection(edge.data_structure, "fit_interval", fit_region, "interval", parent=eels_data_item))
    document_model.append_connection(Connection.PropertyConnection(edge.data_structure, "signal_interval", signal_region, "interval", parent=eels_data_item))

    computation = document_model.create_computation()
    computation.create_input_item("eels_xdata", Symbolic.make_item(model_data_item, type="xdata"))
    computation.create_input_item("region", Symbolic.make_item(pick_region))
    computation.create_input_item("fit_interval", Symbolic.make_item(edge.data_structure), property_name="fit_interval")
    computation.create_input_item("signal_interval", Symbolic.make_item(edge.data_structure), property_name="signal_interval")
    computation.processing_id = "eels.background_subtraction11"
    computation.create_output_item("data", Symbolic.make_item(eels_data_item))
    document_model.append_computation(computation)

    # the eels item will need the initial computation results to display properly (view to intervals)
    await document_model.compute_immediate(document_controller.event_loop, computation)

    # ensure computation is deleted when eels is deleted
    computation.source = eels_data_item

    # create an elemental_mapping_edge_ref data structure, owned by the eels data item, with a referenced
    # object pointing to the edge. used for recognizing the eels data item as such.
    data_structure = document_model.create_data_structure(structure_type="elemental_mapping_edge_ref", source=eels_data_item)
    data_structure.set_referenced_object("spectrum_image", model_data_item)
    data_structure.set_referenced_object("edge", edge.data_structure)
    data_structure.set_referenced_object("data", eels_data_item)
    data_structure.set_referenced_object("pick_region", pick_region)
    document_model.append_data_structure(data_structure)

    # display it
    eels_display_item.view_to_intervals(eels_data_item.xdata, [edge.data_structure.fit_interval, edge.data_structure.signal_interval])
    document_controller.show_display_item(eels_display_item)


async def change_edge(document_controller: DocumentController.DocumentController, model_data_item: DataItem.DataItem, eels_data_item: DataItem.DataItem, edge: "ElementalMappingEdge") -> None:
    """Change the eels data item and associated items to display new edge.

    The library will be changed in the following way:
        - the pick region will be renamed
        - the pick data item will connect fit/signal regions to new edge data structure
        - the background subtraction computation will use edge intervals from new edge
        - the pick, background, subtracted, and eels line plot data items will be renamed
        - the eels line plot will connect fit/signal regions to new edge data structure
        - the edge reference will reference the new edge
    """
    document_model = document_controller.document_model
    project = document_model._project

    computation = None  # type: Symbolic.Computation
    for computation_ in document_model.computations:
        if computation_.source == eels_data_item and computation_.processing_id == "eels.background_subtraction11":
            computation = computation_
            break

    edge_ref_data_structure = None
    old_edge_data_structure = None
    for data_structure in document_model.data_structures:
        if data_structure.source == eels_data_item and data_structure.structure_type == "elemental_mapping_edge_ref":
            edge_ref_data_structure = data_structure
            old_edge_data_structure = data_structure.get_referenced_object("edge")
            break

    if not computation or not edge_ref_data_structure or not old_edge_data_structure:
        return

    pick_region = edge_ref_data_structure.get_referenced_object("pick_region")

    if not eels_data_item or not pick_region:
        return

    pick_region.label = "{} {}".format(_("Pick"), str(edge.electron_shell))

    for connection in copy.copy(document_model.connections):
        if connection.parent == eels_data_item and connection.source_property in ("fit_interval", "signal_interval"):
            source_property = connection.source_property
            target_property = connection.target_property
            target = connection._target
            document_model.remove_connection(connection)
            new_connection = Connection.PropertyConnection(edge.data_structure, source_property, target, target_property, parent=eels_data_item)
            document_model.append_connection(new_connection)

    computation.set_input_item("fit_interval", Symbolic.make_item(edge.data_structure))
    computation.set_input_item("signal_interval", Symbolic.make_item(edge.data_structure))

    eels_data_item.title = "{} EELS Data of {}".format(pick_region.label, model_data_item.title)

    for connection in copy.copy(document_model.connections):
        if connection.parent == eels_data_item and connection.source_property in ("fit_interval", "signal_interval"):
            source_property = connection.source_property
            target_property = connection.target_property
            target = connection._target
            document_model.remove_connection(connection)
            new_connection = Connection.PropertyConnection(edge.data_structure, source_property, target, target_property, parent=eels_data_item)
            document_model.append_connection(new_connection)

    edge_ref_data_structure.remove_referenced_object("edge")
    edge_ref_data_structure.set_referenced_object("edge", edge.data_structure)

    # the eels item will need the initial computation results to display properly (view to intervals)
    await document_model.compute_immediate(document_controller.event_loop, computation)
    eels_display_item = document_model.get_display_item_for_data_item(eels_data_item)
    eels_display_item.view_to_intervals(eels_data_item.xdata, [edge.fit_interval, edge.signal_interval])


class EELSMapping:
    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, **kwargs):
        spectrum_image_xdata = kwargs["spectrum_image_xdata"]
        fit_interval = kwargs["fit_interval"]
        signal_interval = kwargs["signal_interval"]
        atomic_number = kwargs.get("atomic_number")
        shell_number = kwargs.get("shell_number")
        subshell_index = kwargs.get("subshell_index")
        electron_shell = None
        if atomic_number is not None and shell_number is not None and subshell_index is not None:
            electron_shell = PeriodicTable.ElectronShell(atomic_number, shell_number, subshell_index)
        self.__mapped_xdata = eels_analysis.map_background_subtracted_signal(spectrum_image_xdata, electron_shell, [fit_interval], signal_interval)

    def commit(self):
        self.computation.set_referenced_xdata("map", self.__mapped_xdata)


async def map_new_edge(document_controller, model_data_item, edge) -> None:
    document_model = document_controller.document_model
    project = document_model._project

    map_data_item = DataItem.new_data_item()
    map_data_item.title = "{} of {}".format(_("Map"), str(edge.electron_shell))
    map_data_item.category = model_data_item.category
    map_data_item.source = model_data_item
    document_model.append_data_item(map_data_item)

    computation = document_model.create_computation()
    computation.source = map_data_item
    computation.create_input_item("spectrum_image_xdata", Symbolic.make_item(model_data_item, type="xdata"))
    computation.create_input_item("fit_interval", Symbolic.make_item(edge.data_structure), property_name="fit_interval")
    computation.create_input_item("signal_interval", Symbolic.make_item(edge.data_structure), property_name="signal_interval")
    computation.create_variable(name="atomic_number", value_type="integral", value=edge.electron_shell.atomic_number)
    computation.create_variable(name="shell_number", value_type="integral", value=edge.electron_shell.shell_number)
    computation.create_variable(name="subshell_index", value_type="integral", value=edge.electron_shell.subshell_index)
    computation.processing_id = "eels.mapping"
    computation.create_output_item("map", Symbolic.make_item(map_data_item))
    document_model.append_computation(computation)

    await document_model.compute_immediate(document_controller.event_loop, computation)

    map_display_item = document_model.get_display_item_for_data_item(map_data_item)
    document_controller.show_display_item(map_display_item)


class ElementalMappingEdge:
    def __init__(self, *, data_structure=None, electron_shell: PeriodicTable.ElectronShell=None, fit_interval=None, signal_interval=None):
        self.__data_structure = data_structure
        self.__fit_interval = fit_interval
        self.__signal_interval = signal_interval
        self.__electron_shell = electron_shell
        if self.__data_structure:
            self.read(self.__data_structure)

    @property
    def data_structure(self):
        return self.__data_structure

    def read(self, data_structure) -> None:
        atomic_number = data_structure.get_property_value("atomic_number")
        shell_number = data_structure.get_property_value("shell_number")
        subshell_index = data_structure.get_property_value("subshell_index")
        self.__electron_shell = PeriodicTable.ElectronShell(atomic_number, shell_number, subshell_index)
        self.__fit_interval = data_structure.get_property_value("fit_interval", (0.4, 0.5))
        self.__signal_interval = data_structure.get_property_value("signal_interval", (0.5, 0.6))

    def write(self, data_structure) -> None:
        self.__write_electron_shell(data_structure)
        self.__write_fit_interval(data_structure)
        self.__write_signal_interval(data_structure)

    def matches(self, data_structure) -> bool:
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
                if self.__is_explorer(document_model, data_item):
                    self.__connect_explorer_interval(document_model, data_item)

        def item_removed(key, value, index):
            if key == "data_item":
                data_item = value
                self.__disconnect_explorer_interval(data_item)

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

        is_explorer = self.__is_explorer(self.__document_model, data_item)
        if is_explorer:
            self.__explorer_interval = self.__energy_intervals.get(data_item.uuid)
        else:
            self.__explorer_interval = None

        self.__model_data_item = None
        self.__edge_data_structure = None

        if self.__is_model(data_item):
            self.__model_data_item = data_item
        elif data_item:
            for data_structure in copy.copy(self.__document_model.data_structures):
                # check to see if the data item is a eels data item with an associated edge. the data item is a
                # eels data item when there is an elemental_mapping_edge_ref with its source being the data item.
                if data_structure.source == data_item and data_structure.structure_type == "elemental_mapping_edge_ref":
                    self.__edge_data_structure = data_structure.get_referenced_object("edge")
                    self.__model_data_item = data_structure.get_referenced_object("spectrum_image")
            if is_explorer:
                self.__model_data_item = data_item.source

    @property
    def model_data_item(self):
        return self.__model_data_item

    @property
    def edge(self):
        return ElementalMappingEdge(data_structure=self.__edge_data_structure) if self.__edge_data_structure else None

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

    def __is_explorer(self, document_model, data_item) -> bool:
        if isinstance(data_item, DataItem.DataItem):
            if data_item.is_data_1d:
                for display_item in document_model.get_display_items_for_data_item(data_item):
                    for graphic in display_item.graphics:
                        if isinstance(graphic, Graphics.IntervalGraphic) and graphic.graphic_id == "explore":
                            return True
        return False

    def __is_calibrated_map(self, data_item) -> bool:
        if isinstance(data_item, DataItem.DataItem):
            if data_item.is_data_2d:
                return data_item.title.startswith("Map") and data_item.intensity_calibration.units.startswith("~")
        return False

    async def explore_edges(self, document_controller):
        document_model = document_controller.document_model
        model_data_item = self.__model_data_item
        model_display_item = document_model.get_display_item_for_data_item(model_data_item)
        pick_region = Graphics.RectangleGraphic()
        pick_region.size = min(1 / 16, 16 / model_data_item.dimensional_shape[0]), min(1 / 16, 16 / model_data_item.dimensional_shape[1])
        pick_region.label = _("Explore")
        model_display_item.add_graphic(pick_region)
        pick_data_item = document_model.get_pick_region_new(model_display_item, model_display_item.data_item, pick_region=pick_region)
        if pick_data_item:
            explore_interval = Graphics.IntervalGraphic()
            explore_interval.interval = 0.4, 0.6
            explore_interval.label = _("Explore")
            explore_interval.graphic_id = "explore"
            pick_data_item.source = model_data_item
            pick_display_item = document_model.get_display_item_for_data_item(pick_data_item)
            pick_display_item.add_graphic(explore_interval)
            document_controller.show_display_item(pick_display_item)
            await self.__document_model.compute_immediate(document_controller.event_loop, document_model.get_data_item_computation(pick_data_item))  # need the data to make connect_explorer_interval work; so do this here. ugh.
            self.__connect_explorer_interval(document_model, pick_data_item)

    def __add_edge(self, data_item, electron_shell, fit_interval, signal_interval) -> ElementalMappingEdge:
        project = self.__document_model._project
        data_structure = self.__document_model.create_data_structure(structure_type="elemental_mapping_edge", source=data_item)
        self.__document_model.append_data_structure(data_structure)
        edge = ElementalMappingEdge(electron_shell=electron_shell, fit_interval=fit_interval, signal_interval=signal_interval)
        edge.write(data_structure)
        return ElementalMappingEdge(data_structure=data_structure)

    def __remove_edge(self, edge: ElementalMappingEdge) -> None:
        for data_structure in copy.copy(self.__document_model.data_structures):
            if data_structure.source == self.__model_data_item and edge.matches(data_structure):
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

    def __connect_explorer_interval(self, document_model, data_item):
        if data_item.is_data_1d:
            for display_item in document_model.get_display_items_for_data_item(data_item):
                for graphic in display_item.graphics:
                    if isinstance(graphic, Graphics.IntervalGraphic) and graphic.graphic_id == "explore":
                        dimensional_shape = data_item.dimensional_shape
                        dimensional_calibrations = data_item.dimensional_calibrations
                        self.__explorer_property_changed_listeners[data_item.uuid] = graphic.property_changed_event.listen(functools.partial(self.graphic_property_changed, graphic, data_item, dimensional_shape, dimensional_calibrations))
                        self.graphic_property_changed(graphic, data_item, dimensional_shape, dimensional_calibrations, "interval")

    def __disconnect_explorer_interval(self, data_item):
        listener = self.__explorer_property_changed_listeners.get(data_item.uuid)
        if listener:
            listener.close()
            del self.__explorer_property_changed_listeners[data_item.uuid]

    def add_edge(self, electron_shell: PeriodicTable.ElectronShell) -> typing.Optional[ElementalMappingEdge]:
        model_data_item = self.__model_data_item
        if model_data_item:
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
                    return self.__add_edge(model_data_item, electron_shell, fit_interval, signal_interval)
        return None

    def remove_edge(self, edge: ElementalMappingEdge) -> None:
        self.__remove_edge(edge)

    def build_edge_bundles(self, document_controller):
        document_model = self.__document_model
        model_data_item = self.__model_data_item
        current_data_item = self.__current_data_item
        edge_data_structure = self.__edge_data_structure

        EdgeBundle = collections.namedtuple("EdgeBundle", ["electron_shell_str", "selected", "select_action", "pick_action", "map_action", "delete_action"])

        edge_bundles = list()

        edges = list()
        for data_structure in copy.copy(document_model.data_structures):
            if data_structure.source == model_data_item and data_structure.structure_type == "elemental_mapping_edge":
                edge = ElementalMappingEdge(data_structure=data_structure)
                edges.append(edge)

        for index, edge in enumerate(edges):

            def change_edge_action(edge):
                document_controller.event_loop.create_task(change_edge(document_controller, model_data_item, current_data_item, edge))

            def pick_edge_action(edge):
                document_controller.event_loop.create_task(pick_new_edge(document_controller, model_data_item, edge))

            def map_edge_action(edge):
                document_controller.event_loop.create_task(map_new_edge(document_controller, model_data_item, edge))

            def delete_edge_action(edge):
                self.__remove_edge(edge)

            edge_bundle = EdgeBundle(electron_shell_str=edge.electron_shell.to_long_str(),
                                     selected=edge.data_structure == edge_data_structure,
                                     select_action=functools.partial(change_edge_action, edge),
                                     pick_action=functools.partial(pick_edge_action, edge),
                                     map_action=functools.partial(map_edge_action, edge),
                                     delete_action=functools.partial(delete_edge_action, edge))

            edge_bundles.append(edge_bundle)

        return edge_bundles

    def build_multiprofile(self, document_controller):
        document_model = document_controller.document_model
        model_data_item = self.__model_data_item
        if not model_data_item:
            return
        project = document_model._project
        multiprofile_display_item = None
        line_profile_regions = list()

        colors = ("rgba(0, 0, 255, 0.5)", "rgba(255, 0, 0, 0.5)", "rgba(0, 255, 0, 0.5)")

        for index, dependent_data_item in enumerate(document_model.get_dependent_data_items(model_data_item)):
            if self.__is_calibrated_map(dependent_data_item):
                dependent_display_item = document_model.get_display_item_for_data_item(dependent_data_item)
                if not multiprofile_display_item:
                    multiprofile_display_item = DisplayItem.DisplayItem()
                    multiprofile_display_item.title = _("Multi-profile")
                    multiprofile_display_item.display_type = "line_plot"
                    multiprofile_display_item.set_display_property("legend_position", "top-right")
                    document_model.append_display_item(multiprofile_display_item)
                line_profile_data_item = document_model.get_line_profile_new(dependent_display_item, dependent_display_item.data_item)
                line_profile_display_item = document_model.get_display_item_for_data_item(line_profile_data_item)
                line_profile_display_data_channel = line_profile_display_item.get_display_data_channel_for_data_item(line_profile_data_item)
                line_profile_region = dependent_display_item.graphics[0]
                line_profile_region.vector = (0.5, 0.2), (0.5, 0.8)
                line_profile_regions.append(line_profile_region)
                multiprofile_display_item.append_display_data_channel_for_data_item(line_profile_data_item)
                display_layers = multiprofile_display_item.display_layers
                display_layers[-1]["label"] = dependent_data_item.title[dependent_data_item.title.index(" of ") + 4:]
                display_layers[-1]["fill_color"] = colors[index % len(colors)]
                multiprofile_display_item.display_layers = display_layers
        if multiprofile_display_item:
            for line_profile_region in line_profile_regions[1:]:
                document_model.append_connection(Connection.PropertyConnection(line_profile_regions[0], "vector", line_profile_region, "vector", parent=multiprofile_display_item))
                document_model.append_connection(Connection.PropertyConnection(line_profile_regions[0], "width", line_profile_region, "width", parent=multiprofile_display_item))
            document_controller.show_display_item(multiprofile_display_item)

Symbolic.register_computation_type("eels.background_subtraction11", EELSBackgroundSubtraction)
Symbolic.register_computation_type("eels.mapping", EELSMapping)
