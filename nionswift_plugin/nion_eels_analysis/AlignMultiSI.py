import typing
import numpy
from nion.data import Core
from nion.data import DataAndMetadata
from nion.swift.model import Symbolic
from nion.typeshed import API_1_0
from nion.utils import Event


class AlignMultiSI:
    def __init__(self, computation, **kwargs):
        self.computation = computation
        self.progress_updated_event = Event.Event()

        def create_panel_widget(ui, document_controller):
            def update_align_region_label():
                current_region = self.computation._computation.get_input("align_region")
                haadf_sequence_data_item = self.computation._computation.get_input("haadf_sequence_data_item")
                if current_region and haadf_sequence_data_item:
                    bounds = current_region.bounds
                    shape = haadf_sequence_data_item.xdata.datum_dimension_shape
                    current_region = ((int(bounds[0][0]*shape[0]), int(bounds[0][1]*shape[1])),
                                      (int(bounds[1][0]*shape[0]), int(bounds[1][1]*shape[1])))
                self.align_region_label.text = str(current_region)

            def select_button_clicked():
                for variable in self.computation._computation.variables:
                    if variable.name == "align_region":
                        self.computation._computation.remove_variable(variable)
                graphics = document_controller.target_display.selected_graphics or list()
                align_region = None
                for graphic in graphics:
                    if graphic.graphic_type == "rect-graphic":
                        align_region = graphic
                        break
                if align_region:
                    self.computation._computation.create_input_item("align_region", Symbolic.make_item(align_region._graphic))

            def align_index_finished(text):
                try:
                    index = int(text)
                except ValueError:
                    current_index = self.computation._computation.get_input_value("align_index") or 0
                    self.index_field.text = str(current_index)
                else:
                    self.computation.set_input_value("align_index", index)

            column = ui.create_column_widget()
            row = ui.create_row_widget()

            select_graphics_button = ui.create_push_button_widget("Select align region")
            self.align_region_label = ui.create_label_widget()
            update_align_region_label()
            row.add_spacing(10)
            row.add(select_graphics_button)
            row.add_spacing(5)
            row.add(ui.create_label_widget("Current region: "))
            row.add(self.align_region_label)
            row.add_stretch()
            row.add_spacing(10)

            index_label = ui.create_label_widget("Reference slice index: ")
            current_index = self.computation._computation.get_input_value("align_index") or 0
            self.index_field = ui.create_line_edit_widget(str(current_index))
            self.index_field.on_editing_finished = align_index_finished
            index_row = ui.create_row_widget()
            index_row.add_spacing(10)
            index_row.add(index_label)
            index_row.add(self.index_field)
            index_row.add_spacing(10)
            index_row.add_stretch()

            column.add_spacing(10)
            column.add(row)
            column.add_spacing(5)
            column.add(index_row)

            select_graphics_button.on_clicked = select_button_clicked

            return column

        self.computation._computation.create_panel_widget = create_panel_widget
        self.computation._computation.progress_updated_event = self.progress_updated_event

    def execute(self, si_sequence_data_item: API_1_0.DataItem, haadf_sequence_data_item: API_1_0.DataItem,
                align_index: int, align_region: typing.Optional[API_1_0.Graphic]=None):
        haadf_xdata = haadf_sequence_data_item.xdata
        si_xdata = si_sequence_data_item.xdata
        bounds = None
        if align_region:
            bounds = align_region.bounds
        translations = Core.function_sequence_measure_relative_translation(haadf_xdata,
                                                                           haadf_xdata[align_index],
                                                                           10, True, bounds=bounds)
        sequence_shape = haadf_sequence_data_item.xdata.sequence_dimension_shape
        data_zeros = (0,) * si_xdata.datum_dimension_count
        c = int(numpy.product(sequence_shape))
        haadf_result_data = numpy.empty_like(haadf_xdata.data)
        si_result_data = numpy.empty_like(si_xdata.data)
        for i in range(c):
            ii = numpy.unravel_index(i, sequence_shape)
            current_xdata = DataAndMetadata.new_data_and_metadata(haadf_xdata.data[ii])
            translation = translations.data[ii]
            haadf_result_data[ii] = Core.function_shift(current_xdata, tuple(translation)).data
            current_xdata = DataAndMetadata.new_data_and_metadata(si_xdata.data[ii])
            si_result_data[ii] = Core.function_shift(current_xdata, tuple(translation) + data_zeros).data
            self.progress_updated_event.fire(0, c, i+1)

        self.__aligned_haadf_sequence = DataAndMetadata.new_data_and_metadata(haadf_result_data,
                                                                              intensity_calibration=haadf_xdata.intensity_calibration,
                                                                              dimensional_calibrations=haadf_xdata.dimensional_calibrations,
                                                                              metadata=haadf_xdata.metadata,
                                                                              data_descriptor=haadf_xdata.data_descriptor)
        self.__aligned_si_sequence = DataAndMetadata.new_data_and_metadata(si_result_data,
                                                                           intensity_calibration=si_xdata.intensity_calibration,
                                                                           dimensional_calibrations=si_xdata.dimensional_calibrations,
                                                                           metadata=si_xdata.metadata,
                                                                           data_descriptor=si_xdata.data_descriptor)

    def commit(self):
        self.computation.set_referenced_xdata("aligned_haadf", self.__aligned_haadf_sequence)
        self.computation.set_referenced_xdata("aligned_si", self.__aligned_si_sequence)


def align_multi_si(api: API_1_0.API, window: API_1_0.DocumentWindow):
    selected_display_items = window._document_controller._get_two_data_sources()
    error_msg = "Select a sequence of spectrum images and a sequence of scanned images in order to use this computation."
    assert selected_display_items[0][0] is not None, error_msg
    assert selected_display_items[1][0] is not None, error_msg
    assert selected_display_items[0][0].data_item is not None, error_msg
    assert selected_display_items[1][0].data_item is not None, error_msg
    assert selected_display_items[0][0].data_item.is_sequence, error_msg
    assert selected_display_items[1][0].data_item.is_sequence, error_msg

    if selected_display_items[0][0].data_item.is_collection:
        si_sequence_data_item = selected_display_items[0][0].data_item
        haadf_sequence_data_item = selected_display_items[1][0].data_item
        align_region = selected_display_items[1][1]
        align_index = selected_display_items[1][0].display_data_channel.sequence_index
    elif selected_display_items[1][0].data_item.is_collection:
        si_sequence_data_item = selected_display_items[1][0].data_item
        haadf_sequence_data_item = selected_display_items[0][0].data_item
        align_region = selected_display_items[0][1]
        align_index = selected_display_items[0][0].display_data_channel.sequence_index
    else:
        raise ValueError(error_msg)

    aligned_haadf = api.library.create_data_item_from_data(numpy.zeros((1,1,1)), title="Aligned {}".format(haadf_sequence_data_item.title))
    aligned_si = api.library.create_data_item_from_data(numpy.zeros((1,1,1)), title="Aligned {}".format(si_sequence_data_item.title))
    inputs = {"si_sequence_data_item": api._new_api_object(si_sequence_data_item),
              "haadf_sequence_data_item": api._new_api_object(haadf_sequence_data_item),
              "align_index": align_index}
    if align_region:
        inputs["align_region"] = api._new_api_object(align_region)
    computation = api.library.create_computation("eels.align_multi_si",
                                                 inputs=inputs,
                                                 outputs={"aligned_haadf": aligned_haadf,
                                                          "aligned_si": aligned_si})
    computation._computation.source = aligned_si._data_item
    window.display_data_item(aligned_haadf)
    window.display_data_item(aligned_si)

Symbolic.register_computation_type("eels.align_multi_si", AlignMultiSI)