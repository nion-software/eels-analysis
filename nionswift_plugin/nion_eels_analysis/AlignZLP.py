# imports
import logging
import copy
import numpy
import typing
import contextlib
import scipy.ndimage

# local libraries
from nion.data import DataAndMetadata
from nion.eels_analysis import ZLP_Analysis
from nion.swift import Facade
from nion.typeshed import API_1_0 as API
from nion.ui import Declarative
from nion.ui import Dialog
from nion.ui import Window
from nion.utils import Converter
from nion.utils import Event

DataArrayType = numpy.typing.NDArray[typing.Any]


def align_zlp_xdata(src_xdata: DataAndMetadata.DataAndMetadata,
                    progress_fn: typing.Optional[typing.Callable[[int], None]] = None, method: str = 'com',
                    roi: typing.Optional[Facade.Graphic] = None, ref_index: int = 0) -> typing.Tuple[typing.Optional[DataAndMetadata.DataAndMetadata], typing.Optional[DataAndMetadata.DataAndMetadata]]:
    # check to make sure it is suitable for this algorithm
    # if (src_xdata.is_datum_1d and (src_xdata.is_sequence or src_xdata.is_collection)) or (src_xdata.is_datum_2d and not (src_xdata.is_sequence or src_xdata.is_collection)):
    if src_xdata.is_datum_1d or (src_xdata.is_datum_2d and not (src_xdata.is_sequence or src_xdata.is_collection)):
        # get the numpy array and create the destination data
        src_data = src_xdata.data
        assert src_data is not None

        d_rank = 1
        src_shape = tuple(src_xdata.data_shape)
        d_shape = src_shape[-d_rank:]

        if roi and roi.graphic_type == "interval-graphic":
            data_slice = slice(int(typing.cast(float, roi.start) * d_shape[0]), int(typing.cast(float, roi.end) * d_shape[0]))
        elif roi and roi.graphic_type == "rect-graphic":
            ref_index = int(roi.bounds[0][0] * src_shape[0])
            data_slice = slice(int(roi.bounds[0][1] * d_shape[0]), int((roi.bounds[0][1] + roi.bounds[1][1]) * d_shape[0]))
        else:
            data_slice = slice(0, None)

        flat_src_data = numpy.reshape(src_data, (-1,) + d_shape)
        flat_dst_data = numpy.zeros_like(flat_src_data)
        flat_pos_data: DataArrayType = numpy.zeros(flat_src_data.shape[0], dtype=numpy.float32)

        get_position_fn: typing.Callable[[DataArrayType], typing.Tuple[typing.Optional[float], ...]]
        if method == "com":
            get_position_fn = ZLP_Analysis.estimate_zlp_amplitude_position_width_com
            interpolation_order = 1
        elif method == "fit":
            get_position_fn = ZLP_Analysis.estimate_zlp_amplitude_position_width_fit_spline
            interpolation_order = 1
        elif method == "max":
            get_position_fn = lambda data: (None, typing.cast(float, numpy.argmax(data)), None)
            interpolation_order = 0
        else:
            raise ValueError(f"Method {method} is not supported. Allowed options are 'com', 'fit' and 'max'.")

        # use this as the reference position. all other spectra will be aligned to this one.
        ref_pos = get_position_fn(flat_src_data[ref_index, data_slice])[1] or 0.0
        # put the first spectrum in the result
        flat_dst_data[ref_index] = flat_src_data[ref_index]
        # loop over all non-datum dimensions linearly
        for i in range(len(flat_src_data)):
            if i == ref_index:
                continue
            # the algorithm in this early version is to find the max value
            mx_pos = get_position_fn(flat_src_data[i, data_slice])[1] or 0.0
            # fallback to simple max if get_position_fun failed
            if mx_pos is numpy.nan:
                mx_pos = typing.cast(float, numpy.argmax(flat_src_data[i, data_slice]))
            # determine the offset and apply it
            offset = ref_pos - mx_pos
            flat_dst_data[i] = scipy.ndimage.shift(flat_src_data[i], offset, order=interpolation_order)
            flat_pos_data[i] = -offset
            # every row, report progress (will also work for a sequence or 1d collection
            # because there we have only 1 row anyways)
            if i % src_shape[1] == 0 and callable(progress_fn):
                progress_fn(i//src_shape[1])

        dimensional_calibrations = copy.deepcopy(src_xdata.dimensional_calibrations)
        energy_calibration = dimensional_calibrations[-1]
        energy_calibration.offset = -(ref_pos + 0.5) * energy_calibration.scale
        dimensional_calibrations = list(dimensional_calibrations[:-1]) + [energy_calibration]
        shift_calibration = copy.copy(energy_calibration)
        shift_calibration.offset = 0

        # dst_data is complete. construct xdata with correct calibration and data descriptor.
        data_descriptor = DataAndMetadata.DataDescriptor(src_xdata.is_sequence, src_xdata.collection_dimension_count, src_xdata.datum_dimension_count)
        shift_xdata = None
        if flat_pos_data.size > 1:
            shift_xdata = DataAndMetadata.new_data_and_metadata(flat_pos_data.reshape(src_shape[:-d_rank]), shift_calibration, dimensional_calibrations[:-d_rank])
        return (DataAndMetadata.new_data_and_metadata(flat_dst_data.reshape(src_shape), src_xdata.intensity_calibration, dimensional_calibrations, data_descriptor=data_descriptor),
                shift_xdata)

    return None, None


def _run_align_zlp(api: API.API, window: Facade.DocumentWindow, method_id: str, method_name: str) -> None:
    # find the focused data item
    src_display = window.target_display
    if src_display and src_display.data_item:
        assert src_display.data_item.xdata

        ref_index = 0
        if src_display._display_item.display_data_channel:
            # Using the sequence index as reference only makes sense for "pure" sequences because the index will be
            # interpreted as index in the flattened non-datum axes
            if src_display.data_item.xdata.is_sequence and not src_display.data_item.xdata.is_collection:
                ref_index = src_display._display_item.display_data_channel.sequence_index

        def progress(i: int) -> None:
            logging.info(f"Processing row {i} (align zlp)")

        roi = src_display.selected_graphics[0] if src_display.selected_graphics else None
        dst_xdata, shift_xdata = align_zlp_xdata(src_display.data_item.xdata, progress, method=method_id, roi=roi, ref_index=ref_index)

        if dst_xdata:
            # create a new data item in the library and set its title.
            if shift_xdata:
                shift_data_item = api.library.create_data_item_from_data_and_metadata(shift_xdata)
                shift_data_item.title = f"Shifts ({method_name}) " + src_display._display_item.displayed_title
            data_item = api.library.create_data_item_from_data_and_metadata(dst_xdata)
            data_item.title = f"Aligned ({method_name}) " + src_display._display_item.displayed_title

            # display the data item.
            window.display_data_item(data_item)
        else:
            logging.error("Failed: Data is not a sequence or collection of 1D spectra.")
    else:
        logging.error("Failed: No data item selected.")


def align_zlp(api: API.API, window: Facade.DocumentWindow) -> None:
    _run_align_zlp(api, window, "max", "max")


def align_zlp_com(api: API.API, window: Facade.DocumentWindow) -> None:
    _run_align_zlp(api, window, "com", "com")


def align_zlp_fit(api: API.API, window: Facade.DocumentWindow) -> None:
    _run_align_zlp(api, window, "fit", "peak fit")


def _calibrate_spectrum(api: Facade.API_1, window: Facade.DocumentWindow) -> typing.Optional[Dialog.ActionDialog]:

    class UIHandler(Declarative.Handler):
        def __init__(self, data_item: Facade.DataItem, src_data_item: Facade.DataItem, offset_graphic: Facade.Graphic,
                     second_graphic: Facade.Graphic, units: str = 'eV') -> None:
            super().__init__()
            assert data_item.display_xdata
            self.ev_converter = Converter.PhysicalValueToStringConverter(units)
            self.property_changed_event = Event.Event()
            self.__data_item = data_item
            self.__src_data_item = src_data_item
            self.__offset_graphic = offset_graphic
            self.__second_graphic = second_graphic
            self.__offset_energy = 0.0
            self.__graphic_updating = False
            self.__second_point = data_item.display_xdata.dimensional_calibrations[0].convert_to_calibrated_value(typing.cast(float, second_graphic.position) * len(data_item.display_xdata.data))
            self.__offset_changed_listener = offset_graphic._graphic.property_changed_event.listen(self.__offset_graphic_changed)
            self.__second_changed_listener = second_graphic._graphic.property_changed_event.listen(self.__second_graphic_changed)

        def close(self) -> None:
            self.__offset_changed_listener.close()
            self.__offset_changed_listener = typing.cast(typing.Any, None)
            self.__second_changed_listener.close()
            self.__second_changed_listener = typing.cast(typing.Any, None)
            self.__data_item = typing.cast(typing.Any, None)
            self.__src_data_item = typing.cast(typing.Any, None)
            self.__second_graphic = typing.cast(typing.Any, None)
            self.__offset_graphic = typing.cast(typing.Any, None)
            super().close()

        @property
        def offset_energy(self) -> float:
            return self.__offset_energy

        @offset_energy.setter
        def offset_energy(self, offset_energy: float) -> None:
            self.__offset_energy = offset_energy
            self.property_changed_event.fire("offset_energy")
            self.__update_calibration(keep_scale=True)

        @property
        def second_point(self) -> float:
            return self.__second_point

        @second_point.setter
        def second_point(self, energy: float) -> None:
            self.__second_point = energy
            self.property_changed_event.fire("second_point")
            self.__update_calibration()

        @contextlib.contextmanager
        def __lock_graphic_updates(self) -> typing.Iterator[bool]:
            self.__graphic_updating = True
            try:
                yield self.__graphic_updating
            finally:
                self.__graphic_updating = False

        def __update_calibration(self, keep_scale: bool = False) -> None:
            assert self.__data_item.display_xdata
            dimensional_calibrations = copy.deepcopy(self.__data_item.display_xdata.dimensional_calibrations)
            energy_calibration = dimensional_calibrations[0]
            if keep_scale:
                scale = energy_calibration.scale
            else:
                scale = (self.__second_point - self.__offset_energy) / ((typing.cast(float, self.__second_graphic.position) - typing.cast(float, self.__offset_graphic.position)) * len(self.__data_item.display_xdata.data))
            offset = self.__offset_energy - typing.cast(float, self.__offset_graphic.position) * len(self.__data_item.display_xdata.data) * scale
            energy_calibration.scale = scale
            energy_calibration.offset = offset
            assert self.__src_data_item.xdata
            dimensional_calibrations = list(self.__src_data_item.xdata.dimensional_calibrations)
            dimensional_calibrations[-1] = energy_calibration
            self.__src_data_item.set_dimensional_calibrations(dimensional_calibrations)
            offset_graphic_position = (self.offset_energy - offset) / scale / len(self.__data_item.display_xdata.data)
            second_graphic_position = (self.second_point - offset) / scale / len(self.__data_item.display_xdata.data)
            with self.__lock_graphic_updates():
                # use setattr to get around type checking issues
                setattr(self.__offset_graphic, "position", min(max(0.0, offset_graphic_position), 0.99))
                setattr(self.__second_graphic, "position", min(max(0.0, second_graphic_position), 0.99))

        def __offset_graphic_changed(self, property_name: str) -> None:
            if not self.__graphic_updating:
                self.__update_calibration(keep_scale=True)

        def __second_graphic_changed(self, property_name: str) -> None:
            if not self.__graphic_updating:
                self.__update_calibration()


    ui = Declarative.DeclarativeUI()

    row_1 = ui.create_row(ui.create_label(text="Move the graphics in the spectrum and/or change the numbers\nin the fields below to change the calibration.\n"
                                               "The offset graphic will be positioned on the ZLP if possible."),
                          margin=5, spacing=5)
    row_2 = ui.create_row(ui.create_label(text="Offset Point energy"),
                          ui.create_line_edit(text="@binding(offset_energy, converter=ev_converter)"),
                          ui.create_stretch(),
                          margin=5, spacing=5)
    row_3 = ui.create_row(ui.create_label(text="Scale Point energy"),
                          ui.create_line_edit(text="@binding(second_point, converter=ev_converter)"),
                          ui.create_stretch(),
                          margin=5, spacing=5)
    column = ui.create_column(row_1, row_2, row_3, ui.create_stretch(), margin=5, spacing=5)

    data_item: typing.Optional[Facade.DataItem] = window.target_data_item

    class DummyHandler:
        def close(self) -> None: pass

    if not data_item or not data_item.display_xdata or not data_item.display_xdata.is_data_1d:

        window.show_modeless_dialog(ui.create_modeless_dialog(ui.create_label(text=("This tool cannot be used for the selected type of data.\n"
                                                                                    "To use it you have to select a data item containing 1-D data or a sequence of 1-D data.")),
                                                              title="Calibrate Spectrum", margin=10),
                                    handler=DummyHandler())
        return None

    if data_item._data_item.is_live:

        window.show_modeless_dialog(ui.create_modeless_dialog(ui.create_label(text=("This tool cannot be used on live data.\n"
                                                                                    "To use it you have to select a data item containing 1-D data or a sequence of 1-D data.")),
                                                              title="Calibrate Spectrum", margin=10),
                                    handler=DummyHandler())
        return None

    # This is the data item we will update the calibrations on. If the selected data item is the result of a pick
    # computation we will update the source SI. Otherwise we just update the spectrum itself.
    src_data_item = data_item

    for computation in api.library._document_model.computations:
        if computation.processing_id in {"pick-point", "pick-mask-average", "pick-mask-sum"}:
            if computation.get_output("target") == data_item._data_item:
                input_ = computation.get_input_data_item("src")
                src_data_item = api._new_api_object(input_)

    mx_pos = numpy.nan
    try:
        mx_pos = ZLP_Analysis.estimate_zlp_amplitude_position_width_fit_spline(data_item.display_xdata.data)[1]
    except TypeError:
        pass

    # fallback to com if fit failed
    if mx_pos is numpy.nan:
        mx_pos = ZLP_Analysis.estimate_zlp_amplitude_position_width_com(data_item.display_xdata.data)[1]

    # fallback to simple max if everything else failed
    if mx_pos is numpy.nan:
        mx_pos = float(numpy.argmax(data_item.display_xdata.data))

    # We need to move the detected maximum by half a pixel because we want to center the calibration and the graphic
    # on the pixel center but the maximum is calculated for the left edge.
    mx_pos += 0.5

    dimensional_calibrations = list(data_item.display_xdata.dimensional_calibrations)
    energy_calibration = dimensional_calibrations[0]
    energy_calibration.offset = -mx_pos * energy_calibration.scale
    dimensional_calibrations = list(src_data_item.dimensional_calibrations)
    dimensional_calibrations[-1] = energy_calibration
    src_data_item.set_dimensional_calibrations(dimensional_calibrations)


    offset_graphic = data_item.add_channel_region(mx_pos / len(data_item.display_xdata.data))
    offset_graphic.label = "Offset Point"
    offset_graphic._graphic.color = "#CE00AC"
    second_graphic = data_item.add_channel_region((typing.cast(float, offset_graphic.position) + 1.0) * 0.5)
    second_graphic.label = "Scale Point"
    second_graphic._graphic.color = "#CE00AC"

    handler = UIHandler(data_item, src_data_item, offset_graphic, second_graphic, units=energy_calibration.units)
    dialog = typing.cast(Dialog.ActionDialog,
                         Declarative.construct(window._document_controller.ui, window._document_controller,
                                               ui.create_modeless_dialog(column, title="Calibrate Spectrum"), handler))

    def wc(w: Window.Window) -> None:
        assert data_item
        data_item.remove_region(offset_graphic)
        data_item.remove_region(second_graphic)
        getattr(handler, "configuration_dialog_close_listener").close()
        delattr(handler, "configuration_dialog_close_listener")

    # use set handler to pass type checking.
    setattr(handler, "configuration_dialog_close_listener", dialog._window_close_event.listen(wc))

    dialog.show()

    # Return the dialog which is useful for testing
    return dialog


def calibrate_spectrum(api: Facade.API_1, window: Facade.DocumentWindow) -> None:
    _calibrate_spectrum(api, window)
