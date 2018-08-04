# standard libraries
import collections
import copy
import functools
import gettext
import typing

# third party libraries
import numpy

# local libraries
from nion.swift import DocumentController
from nion.swift.model import Connection
from nion.swift.model import DataItem
from nion.swift.model import DocumentModel
from nion.swift.model import Graphics
from nion.swift.model import Symbolic
from nion.eels_analysis import eels_analysis
from nion.eels_analysis import PeriodicTable

_ = gettext.gettext


class EELSBackgroundSubtraction:
    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, eels_spectrum_xdata, fit_interval, signal_interval):
        signal = eels_analysis.extract_original_signal(eels_spectrum_xdata, [fit_interval], signal_interval)
        self.__background_xdata = eels_analysis.calculate_background_signal(eels_spectrum_xdata, [fit_interval], signal_interval)
        self.__subtracted_xdata = signal - self.__background_xdata

    def commit(self):
        self.computation.set_referenced_xdata("background", self.__background_xdata)
        self.computation.set_referenced_xdata("subtracted", self.__subtracted_xdata)


async def pick_new_edge(document_controller, model_data_item, edge) -> None:
    """Set up a new edge pick from the model data item and the given edge.

    The library will have the following new components and connections:
        - a pick region on the model data item
        - a pick data item with a fit/signal connected to the edge data structure
        - a background subtraction computation with model data item, and edge intervals as inputs
        - a background data item, computed by the background subtraction computation
        - a subtracted data item, computed by the background subtraction computation
        - a composite line plot with pick, background, and subtracted data items as components
        - the composite line plot has fit/signal regions connected to edge data structure
        - the composite line plot owns the pick, background, and subtracted data items
        - the composite line plot owns the computation
        - an edge reference, owned by composite line plot, with reference to edge
        - the edge reference is used to recognize the composite line plot as associated with the referenced edge
    """
    document_model = document_controller.document_model
    pick_region = Graphics.RectangleGraphic()
    pick_region.size = min(1 / 16, 16 / model_data_item.dimensional_shape[0]), min(1 / 16, 16 / model_data_item.dimensional_shape[1])
    pick_region.label = "{} {}".format(_("Pick"), str(edge.electron_shell))
    model_data_item.displays[0].add_graphic(pick_region)

    pick_data_item = document_model.get_pick_region_new(model_data_item, pick_region=pick_region)
    if pick_data_item:
        # set up the pick data item for this edge.
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
        document_model.append_connection(Connection.PropertyConnection(edge.data_structure, "fit_interval", fit_region, "interval", parent=pick_data_item))
        document_model.append_connection(Connection.PropertyConnection(edge.data_structure, "signal_interval", signal_region, "interval", parent=pick_data_item))

        background_data_item = DataItem.DataItem(numpy.zeros(1, ))
        background_data_item.title = "{} Background of {}".format(pick_region.label, model_data_item.title)
        background_display_specifier = DataItem.DisplaySpecifier.from_data_item(background_data_item)
        background_display_specifier.display.display_type = "line_plot"
        document_model.append_data_item(background_data_item)

        subtracted_data_item = DataItem.DataItem(numpy.zeros(1, ))
        subtracted_data_item.title = "{} Subtracted of {}".format(pick_region.label, model_data_item.title)
        subtracted_display_specifier = DataItem.DisplaySpecifier.from_data_item(subtracted_data_item)
        subtracted_display_specifier.display.display_type = "line_plot"
        document_model.append_data_item(subtracted_data_item)

        computation = document_model.create_computation()
        computation.create_object("eels_spectrum_xdata", document_model.get_object_specifier(pick_data_item, "display_xdata"))
        computation.create_input("fit_interval", document_model.get_object_specifier(edge.data_structure), "fit_interval")
        computation.create_input("signal_interval", document_model.get_object_specifier(edge.data_structure), "signal_interval")
        computation.processing_id = "eels.background_subtraction"
        computation.create_result("background", document_model.get_object_specifier(background_data_item, "data_item"))
        computation.create_result("subtracted", document_model.get_object_specifier(subtracted_data_item, "data_item"))
        document_model.append_computation(computation)

        # the composite item will need the initial computation results to display properly (view to intervals)
        await document_model.compute_immediate(document_controller.event_loop, computation)

        composite_data_item = DataItem.CompositeLibraryItem()
        composite_data_item.title = "{} from {}".format(pick_region.label, model_data_item.title)
        composite_data_item.append_data_item(pick_data_item)
        composite_data_item.append_data_item(background_data_item)
        composite_data_item.append_data_item(subtracted_data_item)
        composite_data_item.source = pick_region
        pick_data_item.source = composite_data_item
        subtracted_data_item.source = composite_data_item
        background_data_item.source = composite_data_item
        composite_display_specifier = DataItem.DisplaySpecifier.from_data_item(composite_data_item)
        composite_display_specifier.display.display_type = "line_plot"
        composite_display_specifier.display.dimensional_scales = (model_data_item.dimensional_shape[-1], )
        composite_display_specifier.display.dimensional_calibrations = (model_data_item.dimensional_calibrations[-1], )
        composite_display_specifier.display.intensity_calibration = model_data_item.intensity_calibration
        composite_display_specifier.display.legend_labels = ["Data", "Background", "Signal"]
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
        document_model.append_connection(Connection.PropertyConnection(edge.data_structure, "fit_interval", fit_region, "interval", parent=composite_data_item))
        document_model.append_connection(Connection.PropertyConnection(edge.data_structure, "signal_interval", signal_region, "interval", parent=composite_data_item))
        composite_display_specifier.display.view_to_intervals(pick_data_item.xdata, [edge.fit_interval, edge.signal_interval])
        document_controller.display_data_item(composite_display_specifier)

        # ensure computation is deleted when composite is deleted
        computation.source = composite_data_item

        # create an elemental_mapping_edge_ref data structure, owned by the composite data item, with a referenced
        # object pointing to the edge. used for recognizing the composite data item as such.
        data_structure = document_model.create_data_structure(structure_type="elemental_mapping_edge_ref", source=composite_data_item)
        data_structure.set_referenced_object("spectrum_image", model_data_item)
        data_structure.set_referenced_object("edge", edge.data_structure)
        data_structure.set_referenced_object("pick", pick_data_item)
        data_structure.set_referenced_object("pick_region", pick_region)
        data_structure.set_referenced_object("background", background_data_item)
        data_structure.set_referenced_object("subtracted", subtracted_data_item)
        document_model.append_data_structure(data_structure)


async def change_edge(document_controller: DocumentController.DocumentController, model_data_item: DataItem.DataItem, composite_data_item: DataItem.DataItem, edge: "ElementalMappingEdge") -> None:
    """Change the composite data item and associated items to display new edge.

    The library will be changed in the following way:
        - the pick region will be renamed
        - the pick data item will connect fit/signal regions to new edge data structure
        - the background subtraction computation will use edge intervals from new edge
        - the pick, background, subtracted, and composite line plot data items will be renamed
        - the composite line plot will connect fit/signal regions to new edge data structure
        - the edge reference will reference the new edge
    """
    document_model = document_controller.document_model

    computation = None  # type: Symbolic.Computation
    for computation_ in document_model.computations:
        if computation_.source == composite_data_item and computation_.processing_id == "eels.background_subtraction":
            computation = computation_
            break

    edge_ref_data_structure = None  # type: DocumentModel.DataStructure
    old_edge_data_structure = None  # type: DocumentModel.DataStructure
    for data_structure in document_model.data_structures:
        if data_structure.source == composite_data_item and data_structure.structure_type == "elemental_mapping_edge_ref":
            edge_ref_data_structure = data_structure
            old_edge_data_structure = data_structure.get_referenced_object("edge")
            break

    if not computation or not edge_ref_data_structure or not old_edge_data_structure:
        return

    pick_data_item = edge_ref_data_structure.get_referenced_object("pick")
    pick_region = edge_ref_data_structure.get_referenced_object("pick_region")
    background_data_item = edge_ref_data_structure.get_referenced_object("background")
    subtracted_data_item = edge_ref_data_structure.get_referenced_object("subtracted")

    if not pick_data_item or not pick_region or not background_data_item or not subtracted_data_item:
        return

    pick_region.label = "{} {}".format(_("Pick"), str(edge.electron_shell))

    for connection in copy.copy(document_model.connections):
        if connection.parent == pick_data_item and connection.source_property in ("fit_interval", "signal_interval"):
            source_property = connection.source_property
            target_property = connection.target_property
            target = connection._target
            document_model.remove_connection(connection)
            new_connection = Connection.PropertyConnection(edge.data_structure, source_property, target, target_property, parent=pick_data_item)
            document_model.append_connection(new_connection)

    for computation_variable in computation.variables:
        if computation_variable.name in ("fit_interval", "signal_interval"):
            computation_variable.specifier = document_model.get_object_specifier(edge.data_structure)

    pick_data_item.title = "{} Data of {}".format(pick_region.label, model_data_item.title)
    background_data_item.title = "{} Background of {}".format(pick_region.label, model_data_item.title)
    subtracted_data_item.title = "{} Subtracted of {}".format(pick_region.label, model_data_item.title)
    composite_data_item.title = "{} from {}".format(pick_region.label, model_data_item.title)

    for connection in copy.copy(document_model.connections):
        if connection.parent == composite_data_item and connection.source_property in ("fit_interval", "signal_interval"):
            source_property = connection.source_property
            target_property = connection.target_property
            target = connection._target
            document_model.remove_connection(connection)
            new_connection = Connection.PropertyConnection(edge.data_structure, source_property, target, target_property, parent=composite_data_item)
            document_model.append_connection(new_connection)

    edge_ref_data_structure.remove_referenced_object("edge")
    edge_ref_data_structure.set_referenced_object("edge", edge.data_structure)

    # the composite item will need the initial computation results to display properly (view to intervals)
    await document_model.compute_immediate(document_controller.event_loop, computation)
    composite_display_specifier = DataItem.DisplaySpecifier.from_data_item(composite_data_item)
    composite_display_specifier.display.view_to_intervals(pick_data_item.xdata, [edge.fit_interval, edge.signal_interval])


class EELSMapping:
    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, spectrum_image_xdata, fit_interval, signal_interval):
        self.__mapped_xdata = eels_analysis.map_background_subtracted_signal(spectrum_image_xdata, None, [fit_interval], signal_interval)

    def commit(self):
        self.computation.set_referenced_xdata("map", self.__mapped_xdata)


async def map_new_edge(document_controller, model_data_item, edge) -> None:
    document_model = document_controller.document_model

    map_data_item = DataItem.new_data_item()
    map_data_item.title = "{} of {}".format(_("Map"), str(edge.electron_shell))
    map_data_item.category = model_data_item.category
    map_data_item.source = model_data_item
    document_model.append_data_item(map_data_item)

    computation = document_model.create_computation()
    computation.source = map_data_item
    computation.create_object("spectrum_image_xdata", document_model.get_object_specifier(model_data_item, "xdata"))
    computation.create_input("fit_interval", document_model.get_object_specifier(edge.data_structure), "fit_interval")
    computation.create_input("signal_interval", document_model.get_object_specifier(edge.data_structure), "signal_interval")
    computation.processing_id = "eels.mapping"
    computation.create_result("map", document_model.get_object_specifier(map_data_item, "data_item"))
    document_model.append_computation(computation)

    await document_model.compute_immediate(document_controller.event_loop, computation)

    document_controller.display_data_item(DataItem.DisplaySpecifier.from_data_item(map_data_item))


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
                    self.__connect_explorer_interval(data_item)

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

        is_explorer = self.__is_explorer(data_item)
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
                # check to see if the data item is a composite data item with an associated edge. the data item is a
                # composite data item when there is an elemental_mapping_edge_ref with its source being the data item.
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
        document_model = document_controller.document_model
        model_data_item = self.__model_data_item
        pick_region = Graphics.RectangleGraphic()
        pick_region.size = min(1 / 16, 16 / model_data_item.dimensional_shape[0]), min(1 / 16, 16 / model_data_item.dimensional_shape[1])
        pick_region.label = _("Explore")
        model_data_item.displays[0].add_graphic(pick_region)
        pick_data_item = document_model.get_pick_region_new(model_data_item, pick_region=pick_region)
        if pick_data_item:
            explore_interval = Graphics.IntervalGraphic()
            explore_interval.interval = 0.4, 0.6
            explore_interval.label = _("Explore")
            explore_interval.graphic_id = "explore"
            pick_data_item.source = model_data_item
            pick_display_specifier = DataItem.DisplaySpecifier.from_data_item(pick_data_item)
            pick_display_specifier.display.add_graphic(explore_interval)
            document_controller.display_data_item(pick_display_specifier)
            await self.__document_model.compute_immediate(document_controller.event_loop, document_model.get_data_item_computation(pick_data_item))  # need the data to make connect_explorer_interval work; so do this here. ugh.
            self.__connect_explorer_interval(pick_data_item)

    def __add_edge(self, data_item, electron_shell, fit_interval, signal_interval) -> ElementalMappingEdge:
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

    def __connect_explorer_interval(self, data_item):
        if data_item.is_data_1d:
            for graphic in data_item.displays[0].graphics:
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
        multiprofile_data_item = None
        legend_labels = list()
        line_profile_regions = list()
        for index, dependent_data_item in enumerate(document_model.get_dependent_data_items(model_data_item)):
            if self.__is_calibrated_map(dependent_data_item):
                if not multiprofile_data_item:
                    multiprofile_data_item = DataItem.CompositeLibraryItem()
                    document_model.append_data_item(multiprofile_data_item)
                legend_labels.append(dependent_data_item.title[dependent_data_item.title.index(" of ") + 4:])
                line_profile_data_item = document_model.get_line_profile_new(dependent_data_item)
                line_profile_region = dependent_data_item.displays[0].graphics[0]
                line_profile_region.vector = (0.5, 0.2), (0.5, 0.8)
                multiprofile_data_item.append_data_item(line_profile_data_item)
                line_profile_regions.append(line_profile_region)
        if multiprofile_data_item:
            multiprofile_display_specifier = DataItem.DisplaySpecifier.from_data_item(multiprofile_data_item)
            multiprofile_display_specifier.display.display_type = "line_plot"
            multiprofile_display_specifier.display.dimensional_scales = (model_data_item.dimensional_shape[0], )
            multiprofile_display_specifier.display.dimensional_calibrations = (model_data_item.dimensional_calibrations[0], )
            multiprofile_display_specifier.display.intensity_calibration = model_data_item.intensity_calibration
            multiprofile_display_specifier.display.legend_labels = legend_labels
            for line_profile_region in line_profile_regions[1:]:
                document_model.append_connection(Connection.PropertyConnection(line_profile_regions[0], "vector", line_profile_region, "vector", parent=multiprofile_data_item))
                document_model.append_connection(Connection.PropertyConnection(line_profile_regions[0], "width", line_profile_region, "width", parent=multiprofile_data_item))
            multiprofile_data_item.title = _("Profiles of ") + ", ".join(legend_labels)
            document_controller.display_data_item(multiprofile_display_specifier)

Symbolic.register_computation_type("eels.background_subtraction", EELSBackgroundSubtraction)
Symbolic.register_computation_type("eels.mapping", EELSMapping)
