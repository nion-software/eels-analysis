# standard libraries
import functools
import gettext
import operator

# third party libraries
# None

# local libraries
from nion.swift import Panel
from nion.swift import Workspace
from nion.eels_analysis import PeriodicTable
from nionswift_plugin.nion_eels_analysis import ElementalMappingController

_ = gettext.gettext


class ElementalMappingPanel(Panel.Panel):

    def __init__(self, document_controller, panel_id, properties):
        super().__init__(document_controller, panel_id, _("Elemental Mappings"))

        document_model = document_controller.document_model

        self.__elemental_mapping_controller = ElementalMappingController.ElementalMappingController(document_model)

        ui = document_controller.ui

        self.__button_group = None

        column = ui.create_column_widget()

        edge_column = ui.create_column_widget()

        explore_column = ui.create_column_widget()

        add_edge_column = ui.create_column_widget()

        auto_edge_column = ui.create_column_widget()

        column.add(edge_column)

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

        def data_item_changed(data_item) -> None:
            self.__elemental_mapping_controller.set_current_data_item(data_item)
            current_data_item = data_item
            model_data_item = self.__elemental_mapping_controller.model_data_item
            edge = self.__elemental_mapping_controller.edge
            edge_column.remove_all()
            add_edge_column.remove_all()
            if self.__button_group:
                self.__button_group.close()
                self.__button_group = None
            if model_data_item:
                def explore_pressed():
                    document_controller.event_loop.create_task(self.__elemental_mapping_controller.explore_edges(document_controller))

                explore_button_widget.on_clicked = explore_pressed
                multiprofile_button_widget.on_clicked = functools.partial(self.__elemental_mapping_controller.build_multiprofile, document_controller)
                self.__button_group = ui.create_button_group()
                for index, edge_bundle in enumerate(self.__elemental_mapping_controller.build_edge_bundles(document_controller)):
                    def delete_pressed():
                        edge_bundle.delete_action()
                        data_item_changed(current_data_item)  # TODO: this should be automatic

                    row = ui.create_row_widget()
                    radio_button = None
                    label = None
                    if edge:
                        radio_button = ui.create_radio_button_widget(edge_bundle.electron_shell_str)
                        self.__button_group.add_button(radio_button, index)
                        radio_button.checked = edge_bundle.selected
                        radio_button.on_clicked = edge_bundle.select_action
                    else:
                        label = ui.create_label_widget(edge_bundle.electron_shell_str)
                    delete_button = ui.create_push_button_widget(_("Delete"))
                    pick_button = ui.create_push_button_widget(_("Pick"))
                    map_button = ui.create_push_button_widget(_("Map"))
                    delete_button.on_clicked = delete_pressed
                    pick_button.on_clicked = edge_bundle.pick_action
                    map_button.on_clicked = edge_bundle.map_action
                    row.add_spacing(20)
                    if radio_button:
                        row.add(radio_button)
                        row.add_spacing(4)
                    elif label:
                        row.add(label)
                    row.add_spacing(12)
                    row.add(pick_button)
                    row.add_spacing(12)
                    row.add(map_button)
                    row.add_stretch()
                    row.add(delete_button)
                    row.add_spacing(12)
                    edge_column.add(row)

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
                    self.__elemental_mapping_controller.add_edge(edge_widget.current_item[0])
                    data_item_changed(model_data_item)
                    data_item_changed(data_item)

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
                    explore_interval = self.__elemental_mapping_controller.explorer_interval
                    if explore_interval is not None:
                        edges = PeriodicTable.PeriodicTable().find_edges_in_energy_interval(explore_interval)
                        for i, edge in enumerate(edges[0:4]):
                            button = ui.create_push_button_widget(edge.to_long_str())
                            def add_edge(model_data_item, edge, data_item):
                                self.__elemental_mapping_controller.add_edge(edge)
                                data_item_changed(model_data_item)
                                data_item_changed(data_item)
                            button.on_clicked = functools.partial(add_edge, model_data_item, edge, data_item)
                            col = col1 if i % 2 == 0 else col2
                            col.add(button)
                        col1.add_stretch()
                        col2.add_stretch()

                refresh_widget = ui.create_push_button_widget("\u21BB")
                refresh_widget.on_clicked = lambda: data_item_changed(current_data_item)  # TODO: re-layout in Qt is awful

                update_add_buttons()

                refresh_row.add(refresh_widget)
                refresh_row.add_stretch()

                add_column.add(refresh_row)
                add_column.add(add_grid)

                add_row.add_spacing(20)
                add_row.add(add_column)
                add_row.add_stretch()

                add_edge_column.add(add_row)

        def display_item_changed(display_item):
            data_item = display_item.data_item if display_item else None
            data_item_changed(data_item)

        self.__focused_display_item_changed_event_listener = document_controller.focused_display_item_changed_event.listen(display_item_changed)
        selected_data_item = document_controller.selected_data_item
        data_item_changed(selected_data_item)

    def close(self):
        self.__focused_display_item_changed_event_listener.close()
        self.__focused_display_item_changed_event_listener = None
        if self.__elemental_mapping_controller:
            self.__elemental_mapping_controller.close()
            self.__elemental_mapping_controller = None
        if self.__button_group:
            self.__button_group.close()
            self.__button_group = None
        # continue up the chain
        super().close()

workspace_manager = Workspace.WorkspaceManager()
workspace_manager.register_panel(ElementalMappingPanel, "elemental-mapping-panel", _("Elemental Mappings"), ["left", "right"], "left")
