# standard libraries
import copy
import functools
import gettext
import operator
import typing
import uuid

# third party libraries
# None

# local libraries
import_ok = False
try:
    from nion.utils import Event
    from nion.swift import HistogramPanel
    from nion.swift import Panel
    from nion.swift import Workspace
    from nion.swift.model import Connection
    from nion.swift.model import DataItem
    from nion.swift.model import DocumentModel
    from nion.swift.model import Graphics
    from nion.swift.model import Symbolic
    from EELSAnalysis import PeriodicTable
    import_ok = True
except ImportError:
    pass

_ = gettext.gettext


processing_descriptions = {
    "eels.eels_extract_signal":
        { 'script': 'from EELS import Functions as ea\nfrom nion.data import xdata_1_0 as xd\ntarget.xdata = xd.vstack((ea.extract_signal_from_polynomial_background({src}, signal.interval, (fit.interval, )), {src}))',
          'sources': [{'label': 'Source', 'name': 'src', 'regions': [
              {'name': 'fit', 'params': {'label': 'Fit', 'interval': (0.2, 0.3)}, 'type': 'interval'},
              {'name': 'signal', 'params': {'label': 'Signal', 'interval': (0.4, 0.5)}, 'type': 'interval'},
          ]}],
          'title': 'Background Subtracted',
        },
    "eels.subtract_linear_background":
        { 'script': 'from EELS import Functions as ea\nfrom nion.data import xdata_1_0 as xd\ntarget.xdata = xd.vstack((ea.subtract_linear_background({src}, fit.interval, (0, 1)), {src}))',
          'sources': [{'label': 'Source', 'name': 'src', 'regions': [
              {'name': 'fit', 'params': {'label': 'Fit', 'interval': (0.2, 0.3)}, 'type': 'interval'},
          ]}],
          'title': 'Linear Background Subtracted',
        },
    "eels.subtract_background_signal":
        { 'script': 'from EELS import Functions as ea\nfrom nion.data import xdata_1_0 as xd\nsignal_xdata = ea.extract_original_signal({src}, fit.interval, signal.interval)\nbackground = ea.subtract_background_signal({src}, fit.interval, signal.interval)\ntarget.xdata = xd.vstack((signal_xdata, background, signal_xdata - background))',
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


def explore_edges(document_controller, model_data_item):
    pick_region = Graphics.RectangleGraphic()
    pick_region.size = 16 / model_data_item.maybe_data_source.dimensional_shape[0], 16 / model_data_item.maybe_data_source.dimensional_shape[1]
    pick_region.label = _("Explore")
    model_data_item.maybe_data_source.displays[0].add_graphic(pick_region)
    pick_data_item = document_controller.document_model.make_data_item_with_computation("eels.explore", [(model_data_item, None)], {"src": [pick_region]})
    if pick_data_item:
        new_display_specifier = DataItem.DisplaySpecifier.from_data_item(pick_data_item)
        document_controller.display_data_item(new_display_specifier)
        return pick_data_item
    return None


async def pick_new_edge(document_controller, model_data_item, elemental_mapping) -> None:
    document_model = document_controller.document_model
    pick_region = Graphics.RectangleGraphic()
    pick_region.size = 16 / model_data_item.maybe_data_source.dimensional_shape[0], 16 / model_data_item.maybe_data_source.dimensional_shape[1]
    pick_region.label = "{} {}".format(_("Pick"), str(elemental_mapping.electron_shell))
    model_data_item.maybe_data_source.displays[0].add_graphic(pick_region)
    pick_data_item = document_model.get_pick_region_new(model_data_item, pick_region=pick_region)
    if pick_data_item:
        pick_data_item.title = "{} of {}".format(pick_region.label, model_data_item.title)
        pick_display_specifier = DataItem.DisplaySpecifier.from_data_item(pick_data_item)
        pick_display_specifier.display.display_type = "line_plot"
        pick_display_specifier.display.legend_labels = ["Signal", "Subtracted", "Background"]
        fit_region = Graphics.IntervalGraphic()
        fit_region.label = _("Fit")
        fit_region.graphic_id = "fit"
        fit_region.interval = elemental_mapping.fit_interval
        pick_display_specifier.display.add_graphic(fit_region)
        signal_region = Graphics.IntervalGraphic()
        signal_region.label = _("Signal")
        signal_region.graphic_id = "signal"
        signal_region.interval = elemental_mapping.signal_interval
        pick_display_specifier.display.add_graphic(signal_region)
        # TODO: CHANGES VIA CONNECTIONS DON'T GET WRITTEN TO METADATA
        pick_computation = pick_data_item.maybe_data_source.computation
        pick_computation.create_object("mapping", document_model.get_object_specifier(elemental_mapping), label="Mapping")
        pick_computation.expression = """from EELS import Functions as ea
from nion.data import xdata_1_0 as xd
pick = xd.sum_region(src.xdata, region.mask_xdata_with_shape(src.xdata.data_shape[0:2]))
s = ea.make_signal_like(ea.extract_original_signal(pick, mapping.fit_interval, mapping.signal_interval), pick)
bg = ea.make_signal_like(ea.subtract_background_signal(pick, mapping.fit_interval, mapping.signal_interval), pick)
target.xdata = xd.vstack((pick, s - bg, bg))"""
        pick_data_item.add_connection(Connection.PropertyConnection(elemental_mapping, "fit_interval", fit_region, "interval"))
        pick_data_item.add_connection(Connection.PropertyConnection(elemental_mapping, "signal_interval", signal_region, "interval"))
        await document_controller.document_model.recompute_immediate(document_controller.event_loop, pick_data_item)  # need the data to scale display; so do this here. ugh.
        pick_display_specifier.display.view_to_intervals(pick_data_item.maybe_data_source.data_and_metadata, [elemental_mapping.fit_interval, elemental_mapping.signal_interval])
        document_controller.display_data_item(pick_display_specifier)
    return pick_data_item


def map_new_edge(document_controller, model_data_item, elemental_mapping):
    document_model = document_controller.document_model

    map_data_item = DataItem.new_data_item()
    map_data_item.title = "{} of {}".format(_("Map"), str(elemental_mapping.electron_shell))
    map_data_item.category = model_data_item.category
    document_model.append_data_item(map_data_item)
    display_specifier = DataItem.DisplaySpecifier.from_data_item(map_data_item)
    buffered_data_source = display_specifier.buffered_data_source
    script = "from EELS import Functions as ea\ntarget.xdata = ea.map_background_subtracted_signal(src.xdata, mapping.electron_shell, mapping.fit_interval, mapping.signal_interval)"
    computation = document_model.create_computation(script)
    computation.label = "EELS Map"
    computation.processing_id = "eels.map"
    computation.create_object("src", document_model.get_object_specifier(model_data_item), label="Source")
    computation.create_object("mapping", document_model.get_object_specifier(elemental_mapping), label="Mapping")
    document_model.set_data_item_computation(display_specifier.data_item, computation)

    document_controller.display_data_item(DataItem.DisplaySpecifier.from_data_item(map_data_item))

    return map_data_item


def build_multiprofile(document_controller, model_data_item):
    document_model = document_controller.document_model
    multiprofile_data_item = None
    multiprofile_computation = None
    indexes = list()
    legend_labels = list()
    line_profile_regions = list()
    for index, dependent_data_item in enumerate(document_model.get_dependent_data_items(model_data_item)):
        if is_calibrated_map(dependent_data_item):
            if not multiprofile_data_item:
                multiprofile_data_item = DataItem.new_data_item()
                multiprofile_computation = document_model.create_computation("")
                multiprofile_computation.label = "EELS Multiprofile"
                multiprofile_computation.processing_id = "eels.multiprofile"
            indexes.append(index)
            legend_labels.append(dependent_data_item.title[dependent_data_item.title.index(" of ") + 4:])
            display = dependent_data_item.maybe_data_source.displays[0]
            line_profile_region = Graphics.LineProfileGraphic()
            line_profile_region.start = 0.5, 0.2
            line_profile_region.end = 0.5, 0.8
            display.add_graphic(line_profile_region)
            line_profile_regions.append(line_profile_region)
            multiprofile_computation.create_object("src" + str(index), document_model.get_object_specifier(dependent_data_item), label="Src" + str(index))
            multiprofile_computation.create_object("region" + str(index), document_model.get_object_specifier(line_profile_region), label="Region" + str(index))
    if multiprofile_data_item:
        script = "from EELS import Functions as ea\nfrom nion.data import xdata_1_0 as xd\nimport numpy\n"
        for index in indexes:
            script += "d{0} = xd.line_profile(src{0}.display_xdata, region{0}.vector, region{0}.line_width)\n".format(index)
        profiles = ",".join(["d{0}".format(index) for index in indexes])
        script += "mx=numpy.amax(xd.vstack(({})).data)\n".format(profiles)
        for index in indexes:
            script += "d{0} /= mx\n".format(index)
        script += "target.xdata = xd.vstack(({}))".format(profiles)
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


class ElementalMapping:
    def __init__(self, electron_shell: PeriodicTable.ElectronShell=None, fit_interval=None, signal_interval=None):
        self.uuid = uuid.uuid4()
        self.__fit_interval = fit_interval
        self.__signal_interval = signal_interval
        self.__electron_shell = electron_shell
        self.property_changed_event = Event.Event()

    def close(self):
        pass

    def read_from_dict(self, properties):
        self.uuid = uuid.UUID(properties["uuid"])
        atomic_number = properties.get("atomic_number")
        shell_number = properties.get("shell_number")
        subshell_index = properties.get("subshell_index")
        self.__electron_shell = PeriodicTable.ElectronShell(atomic_number, shell_number, subshell_index)
        self.__fit_interval = properties.get("fit_interval", (0.4, 0.5))
        self.__signal_interval = properties.get("signal_interval", (0.5, 0.6))

    def write_to_dict(self):
        properties = dict()
        properties["type"] = "elemental-mapping"
        properties["uuid"] = str(self.uuid)
        properties["atomic_number"] = self.__electron_shell.atomic_number
        properties["shell_number"] = self.__electron_shell.shell_number
        properties["subshell_index"] = self.__electron_shell.subshell_index
        properties["fit_interval"] = copy.copy(self.__fit_interval)
        properties["signal_interval"] = copy.copy(self.__signal_interval)
        return properties

    def copy_from(self, other):
        self.electron_shell = copy.deepcopy(other.electron_shell)
        self.fit_interval = other.fit_interval
        self.signal_interval = other.signal_interval

    @property
    def electron_shell(self):
        return self.__electron_shell

    @electron_shell.setter
    def electron_shell(self, value):
        if self.__electron_shell != value:
            self.__electron_shell = value
            self.property_changed_event.fire("electron_shell")

    @property
    def fit_interval(self):
        return self.__fit_interval

    @fit_interval.setter
    def fit_interval(self, value):
        if self.__fit_interval != value:
            self.__fit_interval = value
            self.property_changed_event.fire("fit_interval")

    @property
    def signal_interval(self):
        return self.__signal_interval

    @signal_interval.setter
    def signal_interval(self, value):
        if self.__signal_interval != value:
            self.__signal_interval = value
            self.property_changed_event.fire("signal_interval")


elemental_mapping_computation_variable_type = Symbolic.ComputationVariableType('elemental_mapping', "ElementalMapping", ElementalMapping)

Symbolic.ComputationVariable.register_computation_variable_type(elemental_mapping_computation_variable_type)


def is_explorer(data_item):
    if isinstance(data_item, DataItem.DataItem):
        buffered_data_source = data_item.maybe_data_source
        if buffered_data_source and buffered_data_source.is_data_1d:
            for graphic in buffered_data_source.displays[0].graphics:
                if isinstance(graphic, Graphics.IntervalGraphic) and graphic.graphic_id == "explore":
                    return True
    return False


def is_model(data_item):
    if isinstance(data_item, DataItem.DataItem):
        buffered_data_source = data_item.maybe_data_source
        return buffered_data_source and buffered_data_source.is_data_3d
    return False


def is_map(data_item):
    if isinstance(data_item, DataItem.DataItem):
        buffered_data_source = data_item.maybe_data_source
        if buffered_data_source and buffered_data_source.is_data_2d:
            return data_item.title.startswith("Map")
    return False


def is_calibrated_map(data_item):
    if isinstance(data_item, DataItem.DataItem):
        buffered_data_source = data_item.maybe_data_source
        if buffered_data_source and buffered_data_source.is_data_2d:
            return data_item.title.startswith("Map") and buffered_data_source.intensity_calibration.units.startswith("~")
    return False


class ElementalMappingController:
    # only supports properties of elemental_mappings; no more complex structure allowed

    def __init__(self, document_model: DocumentModel.DocumentModel):

        self.explore_interval_changed = Event.Event()

        self.__elemental_mapping_property_changed_listeners = dict()  # typing.Dict[uuid.UUID, Any]
        self.__explore_property_changed_listeners = dict()  # typing.Dict[uuid.UUID, Any]

        self.__energy_intervals = dict()  # typing.Dict[uuid.UUID, typing.Tuple[float, float]]

        def item_inserted(key, value, before_index):
            # when a data item is inserted, construct ElementalMapping objects from the metadata
            # and store the element_mapping list on the data item.
            if key == "data_item":
                data_item = value
                if is_model(data_item):  # TODO: improve handling of metadata in derived items so as to not have to skip this
                    buffered_data_source = data_item.maybe_data_source
                    if buffered_data_source:
                        elemental_mappings = list()
                        metadata = buffered_data_source.metadata
                        elemental_mapping_dicts = metadata.get("elemental_mappings", list())
                        for elemental_mapping_dict in elemental_mapping_dicts:
                            elemental_mapping = ElementalMapping()
                            elemental_mapping.read_from_dict(elemental_mapping_dict)
                            elemental_mappings.append(elemental_mapping)
                            elemental_mapping_computation_variable_type.register_object(elemental_mapping)
                            data_item.persistent_object_context.register(elemental_mapping)  # TODO: check this again
                            self.__elemental_mapping_property_changed_listeners[elemental_mapping.uuid] = elemental_mapping.property_changed_event.listen(lambda k: self.__write_metadata(data_item))
                        setattr(data_item, "elemental_mappings", elemental_mappings)
                if is_explorer(data_item):
                    self.connect_explorer_interval(data_item)

        def item_removed(key, value, index):
            if key == "data_item":
                data_item = value
                if is_model(data_item):  # TODO: improve handling of metadata in derived items so as to not have to skip this
                    for elemental_mapping in getattr(data_item, "elemental_mappings", list()):
                        elemental_mapping.close()
                        elemental_mapping_computation_variable_type.unregister_object(elemental_mapping)
                        self.__elemental_mapping_property_changed_listeners[elemental_mapping.uuid].close()
                        del self.__elemental_mapping_property_changed_listeners[elemental_mapping.uuid]
                    delattr(value, "elemental_mappings")
                listener = self.__explore_property_changed_listeners.get(data_item.uuid)
                if listener:
                    listener.close()
                    del self.__explore_property_changed_listeners[data_item.uuid]

        self.__item_inserted_listener = document_model.item_inserted_event.listen(item_inserted)
        self.__item_removed_listener = document_model.item_removed_event.listen(item_removed)

        for index, data_item in enumerate(document_model.data_items):
            item_inserted("data_item", data_item, index)

        document_model.rebind_computations()

    def close(self):
        self.__item_inserted_listener.close()
        self.__item_inserted_listener = None
        self.__item_removed_listener.close()
        self.__item_removed_listener = None

    def __write_metadata(self, data_item):
        buffered_data_source = data_item.maybe_data_source
        if buffered_data_source:
            metadata = buffered_data_source.metadata
            elemental_mapping_dicts = list()
            for elemental_mapping in getattr(data_item, "elemental_mappings", list()):
                elemental_mapping_dicts.append(elemental_mapping.write_to_dict())
            metadata["elemental_mappings"] = elemental_mapping_dicts
            buffered_data_source.set_metadata(metadata)

    def get_elemental_mappings(self, data_item):
        return getattr(data_item, "elemental_mappings", list())

    def add_elemental_mapping(self, data_item, elemental_mapping):
        # add the elemental_mapping to the list on the data item.
        # then update the metadata to reflect the new list.
        elemental_mappings = self.get_elemental_mappings(data_item)
        assert all(em.uuid != elemental_mapping.uuid for em in elemental_mappings)
        elemental_mappings.append(elemental_mapping)
        elemental_mapping_computation_variable_type.register_object(elemental_mapping)
        self.__elemental_mapping_property_changed_listeners[elemental_mapping.uuid] = elemental_mapping.property_changed_event.listen(lambda k: self.__write_metadata(data_item))
        data_item.persistent_object_context.register(elemental_mapping)  # TODO: check this again
        self.__write_metadata(data_item)

    def remove_elemental_mapping(self, data_item, elemental_mapping):
        # remove element_mapping with matching uuid.
        # then update the metadata to reflect the new list.
        elemental_mappings = self.get_elemental_mappings(data_item)
        assert any(em.uuid == elemental_mapping.uuid for em in elemental_mappings)
        elemental_mappings.remove(elemental_mapping)
        elemental_mapping_computation_variable_type.unregister_object(elemental_mapping)
        self.__elemental_mapping_property_changed_listeners[elemental_mapping.uuid].close()
        del self.__elemental_mapping_property_changed_listeners[elemental_mapping.uuid]
        self.__write_metadata(data_item)

    def graphic_property_changed(self, graphic, data_item, dimensional_shape, dimensional_calibrations, key):
        if key == "interval":
            value = graphic.interval
            ss = value[0] * dimensional_shape[-1]
            ee = value[1] * dimensional_shape[-1]
            s = dimensional_calibrations[-1].convert_to_calibrated_value(ss)
            e = dimensional_calibrations[-1].convert_to_calibrated_value(ee)
            self.__energy_intervals[data_item.uuid] = s, e
            self.explore_interval_changed.fire(data_item, (s, e))

    def get_explorer_interval(self, data_item):
        return self.__energy_intervals.get(data_item.uuid) if data_item else None

    def connect_explorer_interval(self, data_item):
        buffered_data_source = data_item.maybe_data_source
        if buffered_data_source and buffered_data_source.is_data_1d:
            for graphic in buffered_data_source.displays[0].graphics:
                if isinstance(graphic, Graphics.IntervalGraphic) and graphic.graphic_id == "explore":
                    dimensional_shape = buffered_data_source.dimensional_shape
                    dimensional_calibrations = buffered_data_source.dimensional_calibrations
                    self.__explore_property_changed_listeners[data_item.uuid] = graphic.property_changed_event.listen(functools.partial(self.graphic_property_changed, graphic, data_item, dimensional_shape, dimensional_calibrations))
                    self.graphic_property_changed(graphic, data_item, dimensional_shape, dimensional_calibrations, "interval")


async def change_elemental_mapping(event_loop, document_model, model_data_item, data_item, elemental_mapping):
    mapping_computation_variable = None
    pick_region_specifier = None
    computation = data_item.maybe_data_source.computation if data_item else None
    if computation:
        for computation_variable in computation.variables:
            if computation_variable.name == "mapping":
                mapping_computation_variable = computation_variable
            if computation_variable.name == "region":
                pick_region_specifier = computation_variable.specifier
    if mapping_computation_variable:
        mapping_computation_variable.specifier = document_model.get_object_specifier(elemental_mapping)
        for connection in copy.copy(data_item.connections):
            if connection.source_property in ("fit_interval", "signal_interval"):
                source_property = connection.source_property
                target_property = connection.target_property
                target = connection._target
                data_item.remove_connection(connection)
                new_connection = Connection.PropertyConnection(elemental_mapping, source_property, target, target_property)
                data_item.add_connection(new_connection)
        if pick_region_specifier:
            pick_region_value = document_model.resolve_object_specifier(pick_region_specifier)
            if pick_region_value:
                pick_region = pick_region_value.value
                pick_region.label = "{} {}".format(_("Pick"), str(elemental_mapping.electron_shell))
                data_item.title = "{} of {}".format(pick_region.label, model_data_item.title)
        else:
                data_item.title = "{} {} of {}".format(_("Map"), str(elemental_mapping.electron_shell), model_data_item.title)
        document_model.rebind_computations()
    display = data_item.maybe_data_source.displays[0]
    if display.display_type == "line_plot":
        intervals = list()
        for graphic in display.graphics:
            if isinstance(graphic, Graphics.IntervalGraphic) and graphic.graphic_id in ("fit", "signal"):
                intervals.append(graphic.interval)
        await document_model.recompute_immediate(event_loop, data_item)  # need the data to scale display; so do this here. ugh.
        display.view_to_intervals(data_item.maybe_data_source.data_and_metadata, intervals)


class ElementalMappingPanel(Panel.Panel):

    def __init__(self, document_controller, panel_id, properties):
        super().__init__(document_controller, panel_id, _("Elemental Mappings"))

        document_model = document_controller.document_model

        self.__elemental_mapping_panel_controller = ElementalMappingController(document_model)

        ui = document_controller.ui

        self.__button_group = None

        column = ui.create_column_widget()

        elemental_mapping_column = ui.create_column_widget()

        explore_column = ui.create_column_widget()

        add_edge_column = ui.create_column_widget()

        auto_edge_column = ui.create_column_widget()

        column.add(elemental_mapping_column)

        column.add_spacing(12)

        column.add(explore_column)

        column.add_spacing(12)

        column.add(add_edge_column)

        column.add_spacing(12)

        column.add(auto_edge_column)

        column.add_spacing(12)

        column.add_stretch()

        self.widget = column

        explore_row = ui.create_row_widget()

        explore_button_widget = ui.create_push_button_widget(_("Explore"))

        multiprofile_button_widget = ui.create_push_button_widget(_("Multiprofile"))

        explore_row.add(explore_button_widget)
        explore_row.add_spacing(8)
        explore_row.add(multiprofile_button_widget)
        explore_row.add_stretch()

        explore_column.add(explore_row)

        model_data_item_ref = [None]  # type: typing.List[DataItem]
        current_data_item_ref = [None]  # type: typing.List[DataItem]
        explore_interval_ref = [None]  # type: typing.List[typing.Tuple[float, float]]

        def explore_interval_changed(data_item, interval) -> None:
            if data_item and data_item == current_data_item_ref[0] and interval is not None:
                explore_interval_ref[0] = interval
            else:
                explore_interval_ref[0] = None

        self.__explore_interval_changed_listener = self.__elemental_mapping_panel_controller.explore_interval_changed.listen(explore_interval_changed)

        def add_edge(electron_shell: PeriodicTable.ElectronShell, data_item: DataItem.DataItem) -> None:
            model_data_item = model_data_item_ref[0]
            if model_data_item:
                binding_energy_eV = PeriodicTable.PeriodicTable().nominal_binding_energy_ev(electron_shell)
                signal_interval_eV = binding_energy_eV, binding_energy_eV * 1.10
                fit_interval_eV = binding_energy_eV * 0.93, binding_energy_eV * 0.98
                buffered_data_source = model_data_item.maybe_data_source
                if buffered_data_source:
                    dimensional_shape = buffered_data_source.dimensional_shape
                    dimensional_calibrations = buffered_data_source.dimensional_calibrations
                    if dimensional_shape is not None and dimensional_calibrations is not None and len(dimensional_calibrations) > 0:
                        calibration = dimensional_calibrations[-1]
                        if calibration.units == "eV":
                            fit_region_start = calibration.convert_from_calibrated_value(fit_interval_eV[0]) / dimensional_shape[-1]
                            fit_region_end = calibration.convert_from_calibrated_value(fit_interval_eV[1]) / dimensional_shape[-1]
                            signal_region_start = calibration.convert_from_calibrated_value(signal_interval_eV[0]) / dimensional_shape[-1]
                            signal_region_end = calibration.convert_from_calibrated_value(signal_interval_eV[1]) / dimensional_shape[-1]
                            fit_interval = fit_region_start, fit_region_end
                            signal_interval = signal_region_start, signal_region_end
                            elemental_mapping = ElementalMapping(electron_shell, fit_interval, signal_interval)
                            self.__elemental_mapping_panel_controller.add_elemental_mapping(model_data_item, elemental_mapping)
                            data_item_changed(model_data_item)
                            data_item_changed(data_item)

        def data_item_changed(data_item) -> None:
            current_data_item_ref[0] = data_item
            model_data_item = None
            current_elemental_mapping = None
            explore_interval_changed(data_item, self.__elemental_mapping_panel_controller.get_explorer_interval(data_item))
            if is_model(data_item):
                model_data_item = data_item
            else:
                computation = data_item.maybe_data_source.computation if data_item else None
                if computation:
                    for computation_variable in computation.variables:
                        if computation_variable.name == "src":
                            src_data_item_value = document_model.resolve_object_specifier(computation_variable.specifier)
                            src_data_item = src_data_item_value.value if src_data_item_value else None
                            if is_model(src_data_item):
                                model_data_item = src_data_item
                        if computation_variable.name == "mapping":
                            current_elemental_mapping_value = document_model.resolve_object_specifier(computation_variable.specifier)
                            current_elemental_mapping = current_elemental_mapping_value.value if current_elemental_mapping_value else None
            model_data_item_ref[0] = model_data_item
            elemental_mapping_column.remove_all()
            add_edge_column.remove_all()
            if self.__button_group:
                self.__button_group.close()
                self.__button_group = None
            if model_data_item:
                def explore_pressed():
                    async def explore():
                        explore_data_item = explore_edges(document_controller, model_data_item)
                        await document_model.recompute_immediate(document_controller.event_loop, explore_data_item)  # need the data to make connect_explorer_interval work; so do this here. ugh.
                        self.__elemental_mapping_panel_controller.connect_explorer_interval(explore_data_item)
                    document_controller.event_loop.create_task(explore())

                explore_button_widget.on_clicked = explore_pressed
                multiprofile_button_widget.on_clicked = functools.partial(build_multiprofile, document_controller, model_data_item)
                self.__button_group = ui.create_button_group()
                for index, elemental_mapping in enumerate(self.__elemental_mapping_panel_controller.get_elemental_mappings(model_data_item)):
                    row = ui.create_row_widget()
                    radio_button = None
                    label = None
                    electron_shell = elemental_mapping.electron_shell
                    text = electron_shell.to_long_str()
                    if not is_model(current_data_item_ref[0]):
                        radio_button = ui.create_radio_button_widget(text)
                        self.__button_group.add_button(radio_button, index)
                        if elemental_mapping == current_elemental_mapping:
                            radio_button.checked = True
                        def change_elemental_mapping_clicked(elemental_mapping):
                            document_controller.event_loop.create_task(change_elemental_mapping(document_controller.event_loop, document_model, model_data_item, current_data_item_ref[0], elemental_mapping))
                        radio_button.on_clicked = functools.partial(change_elemental_mapping_clicked, elemental_mapping)  # elemental_mapping is in the loop
                    else:
                        label = ui.create_label_widget(text)
                    delete_button = ui.create_push_button_widget(_("Delete"))
                    pick_button = ui.create_push_button_widget(_("Pick"))
                    map_button = ui.create_push_button_widget(_("Map"))
                    def pick_pressed(elemental_mapping):
                        if current_data_item_ref[0] == model_data_item:
                            document_controller.event_loop.create_task(pick_new_edge(document_controller, model_data_item, elemental_mapping))
                    def map_pressed(elemental_mapping):
                        if current_data_item_ref[0] == model_data_item:
                            map_new_edge(document_controller, model_data_item, elemental_mapping)
                    def delete_pressed(elemental_mapping):
                        self.__elemental_mapping_panel_controller.remove_elemental_mapping(model_data_item, elemental_mapping)
                        data_item_changed(current_data_item_ref[0])  # TODO: this should be automatic
                    delete_button.on_clicked = functools.partial(delete_pressed, elemental_mapping)
                    pick_button.on_clicked = functools.partial(pick_pressed, elemental_mapping)
                    map_button.on_clicked = functools.partial(map_pressed, elemental_mapping)
                    row.add_spacing(20)
                    if radio_button:
                        row.add(radio_button)
                        row.add_spacing(4)
                    elif label:
                        row.add(label)
                    if is_model(current_data_item_ref[0]):
                        row.add_spacing(12)
                        row.add(pick_button)
                        row.add_spacing(12)
                        row.add(map_button)
                    row.add_stretch()
                    row.add(delete_button)
                    row.add_spacing(12)
                    elemental_mapping_column.add(row)

                if is_model(current_data_item_ref[0]):

                    atomic_number_widget = ui.create_combo_box_widget(items=PeriodicTable.PeriodicTable().get_elements_list(), item_getter=operator.itemgetter(1))

                    edge_widget = ui.create_combo_box_widget(items=PeriodicTable.PeriodicTable().get_edges_list(1), item_getter=operator.itemgetter(1))

                    add_button_widget = ui.create_push_button_widget(_("Add Edge"))

                    atomic_number_row = ui.create_row_widget()
                    atomic_number_row.add_spacing(20)
                    atomic_number_row.add(ui.create_label_widget(_("Element")))
                    atomic_number_row.add_spacing(8)
                    atomic_number_row.add(atomic_number_widget)
                    atomic_number_row.add_spacing(8)
                    atomic_number_row.add_stretch()

                    edge_row = ui.create_row_widget()
                    edge_row.add_spacing(20)
                    edge_row.add(ui.create_label_widget(_("Edge")))
                    edge_row.add_spacing(8)
                    edge_row.add(edge_widget)
                    edge_row.add_spacing(8)
                    edge_row.add_stretch()

                    add_button_row = ui.create_row_widget()
                    add_button_row.add_spacing(20)
                    add_button_row.add(add_button_widget)
                    add_button_row.add_spacing(8)
                    add_button_row.add_stretch()

                    add_edge_column.add(atomic_number_row)
                    add_edge_column.add(edge_row)
                    add_edge_column.add(add_button_row)

                    def add_edge_current():
                        add_edge(edge_widget.current_item[0], data_item)

                    add_button_widget.on_clicked = add_edge_current

                    def atomic_number_changed(item):
                        edge_widget.items = PeriodicTable.PeriodicTable().get_edges_list(item[0])

                    atomic_number_widget.on_current_item_changed = atomic_number_changed

                add_row = ui.create_row_widget()

                add_column = ui.create_column_widget()

                refresh_row = ui.create_row_widget()

                add_grid = ui.create_row_widget()
                col1 = ui.create_column_widget()
                col2 = ui.create_column_widget()
                add_grid.add(col1)
                add_grid.add(col2)
                add_grid.add_stretch()

                def update_add_buttons():
                    col1.remove_all()
                    col2.remove_all()
                    if explore_interval_ref[0] is not None:
                        edges = PeriodicTable.PeriodicTable().find_edges_in_energy_interval(explore_interval_ref[0])
                        for i, edge in enumerate(edges[0:4]):
                            button = ui.create_push_button_widget(edge.to_long_str())
                            button.on_clicked = functools.partial(add_edge, edge, data_item)
                            col = col1 if i % 2 == 0 else col2
                            col.add(button)
                        col1.add_stretch()
                        col2.add_stretch()

                refresh_widget = ui.create_push_button_widget("\u21BB")
                refresh_widget.on_clicked = lambda: data_item_changed(current_data_item_ref[0])  # TODO: re-layout in Qt is awful

                update_add_buttons()

                refresh_row.add(refresh_widget)
                refresh_row.add_stretch()

                add_column.add(refresh_row)
                add_column.add(add_grid)

                add_row.add_spacing(20)
                add_row.add(add_column)
                add_row.add_stretch()

                add_edge_column.add(add_row)

        self.__target_data_item_stream = HistogramPanel.TargetDataItemStream(document_controller).add_ref()
        self.__listener = self.__target_data_item_stream.value_stream.listen(data_item_changed)
        data_item_changed(self.__target_data_item_stream.value)

    def close(self):
        self.__listener.close()
        self.__listener = None
        self.__target_data_item_stream.remove_ref()
        self.__target_data_item_stream = None
        self.__elemental_mapping_panel_controller.close()
        self.__elemental_mapping_panel_controller = None
        self.__explore_interval_changed_listener.close()
        self.__explore_interval_changed_listener = None
        if self.__button_group:
            self.__button_group.close()
            self.__button_group = None
        # continue up the chain
        super().close()

workspace_manager = Workspace.WorkspaceManager()
workspace_manager.register_panel(ElementalMappingPanel, "elemental-mapping-panel", _("Elemental Mappings"), ["left", "right"], "left")

DocumentModel.DocumentModel.register_processing_descriptions(processing_descriptions)
