# imports
import logging
import copy
import numpy
import typing
import scipy.ndimage

# local libraries
from nion.typeshed import API_1_0
from nion.data import DataAndMetadata
from nion.eels_analysis import ZLP_Analysis


def align_zlp_xdata(src_xdata: DataAndMetadata.DataAndMetadata, progress_fn=None, method='com', roi: typing.Optional[API_1_0.Graphic]=None, ref_index: int=0) -> typing.Tuple[typing.Optional[DataAndMetadata.DataAndMetadata], typing.Optional[DataAndMetadata.DataAndMetadata]]:
    # check to make sure it is suitable for this algorithm
    if (src_xdata.is_datum_1d and (src_xdata.is_sequence or src_xdata.is_collection)) or (src_xdata.is_datum_2d and not (src_xdata.is_sequence or src_xdata.is_collection)):
        # get the numpy array and create the destination data
        src_data = src_xdata.data

        d_rank = 1
        src_shape = tuple(src_xdata.data_shape)
        d_shape = src_shape[-d_rank:]

        if roi and roi.graphic_type == "interval-graphic":
            data_slice = slice(int(roi.start * d_shape[0]), int(roi.end * d_shape[0]))
        elif roi and roi.graphic_type == "rect-graphic":
            ref_index = int(roi.bounds[0][0] * src_shape[0])
            data_slice = slice(int(roi.bounds[0][1] * d_shape[0]), int((roi.bounds[0][1] + roi.bounds[1][1]) * d_shape[0]))
        else:
            data_slice = slice(0, None)

        flat_src_data = numpy.reshape(src_data, (-1,) + d_shape)
        flat_dst_data = numpy.zeros_like(flat_src_data)
        flat_pos_data = numpy.zeros(flat_src_data.shape[0], dtype=numpy.float32)

        if method == "com":
            get_position_fn = ZLP_Analysis.estimate_zlp_amplitude_position_width_com
            interpolation_order = 1
        elif method == "fit":
            get_position_fn = ZLP_Analysis.estimate_zlp_amplitude_position_width_fit_spline
            interpolation_order = 1
        elif method == "max":
            get_position_fn = lambda data: (None, numpy.argmax(data), None)
            interpolation_order = 0
        else:
            raise ValueError(f"Method {method} is not supported. Allowed options are 'com', 'fit' and 'max'.")

        # use this as the reference position. all other spectra will be aligned to this one.
        ref_pos = get_position_fn(flat_src_data[ref_index, data_slice])[1]
        # put the first spectrum in the result
        flat_dst_data[ref_index] = flat_src_data[ref_index]
        # loop over all non-datum dimensions linearly
        for i in range(len(flat_src_data)):
            if i == ref_index:
                continue
            # the algorithm in this early version is to find the max value
            mx_pos = get_position_fn(flat_src_data[i, data_slice])[1]
            # fallback to simple max if get_position_fun failed
            if mx_pos is numpy.nan:
                mx_pos = numpy.argmax(flat_src_data[i, data_slice])
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
        return (DataAndMetadata.new_data_and_metadata(flat_dst_data.reshape(src_shape), src_xdata.intensity_calibration, dimensional_calibrations, data_descriptor=data_descriptor),
                DataAndMetadata.new_data_and_metadata(flat_pos_data.reshape(src_shape[:-d_rank]), shift_calibration, dimensional_calibrations[:-d_rank]))

    return None, None


def _run_align_zlp(api: API_1_0.API, window: API_1_0.DocumentWindow, method_id: str, method_name: str):
    # find the focused data item
    src_display = window.target_display
    if src_display and src_display.data_item:
        ref_index = 0
        if src_display._display_item.display_data_channel:
            # Using the sequence index as reference only makes sense for "pure" sequences because the index will be
            # interpreted as index in the flattened non-datum axes
            if src_display.data_item.xdata.is_sequence and not src_display.data_item.xdata.is_collection:
                ref_index = src_display._display_item.display_data_channel.sequence_index

        def progress(i):
            logging.info(f"Processing row {i} (align zlp)")

        roi = src_display.selected_graphics[0] if src_display.selected_graphics else None
        dst_xdata, shift_xdata = align_zlp_xdata(src_display.data_item.xdata, progress, method=method_id, roi=roi, ref_index=ref_index)

        if dst_xdata:
            # create a new data item in the library and set its title.
            shift_data_item = api.library.create_data_item_from_data_and_metadata(shift_xdata)
            shift_data_item.title = f"Shifts ({method_name}) " + src_display.data_item.title
            data_item = api.library.create_data_item_from_data_and_metadata(dst_xdata)
            data_item.title = f"Aligned ({method_name}) " + src_display.data_item.title

            # display the data item.
            window.display_data_item(data_item)
        else:
            logging.error("Failed: Data is not a sequence or collection of 1D spectra.")
    else:
        logging.error("Failed: No data item selected.")


def align_zlp(api: API_1_0.API, window: API_1_0.DocumentWindow):
    _run_align_zlp(api, window, "max", "max")


def align_zlp_com(api: API_1_0.API, window: API_1_0.DocumentWindow):
    _run_align_zlp(api, window, "com", "com")


def align_zlp_fit(api: API_1_0.API, window: API_1_0.DocumentWindow):
    _run_align_zlp(api, window, "fit", "peak fit")
