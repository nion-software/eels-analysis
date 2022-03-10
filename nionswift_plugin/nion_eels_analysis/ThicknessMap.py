# imports
import numpy
import typing

# local libraries
from nion.data import DataAndMetadata
from nion.swift import Facade
from nion.swift.model import Symbolic


def map_thickness_xdata(src_xdata: DataAndMetadata.DataAndMetadata) -> typing.Optional[DataAndMetadata.DataAndMetadata]:
    # note: uses an extra copy of src_xdata.

    # make indexes array, useful in a few calculations
    indexes = numpy.linspace(0, src_xdata.dimensional_shape[-1], src_xdata.dimensional_shape[-1])

    # find the zero loss peaks by looking for maximum values. get the indexes and the maximum values.
    zlp_indexes = numpy.argmax(src_xdata.data, axis=-1)
    zlp_max_array = numpy.amax(src_xdata.data, axis=-1)

    # create a mask which is 1 to the left of the maximum values and 0 elsewhere
    mask_array = numpy.empty_like(src_xdata.data)
    mask_array[:] = indexes[numpy.newaxis, numpy.newaxis, ...] < zlp_indexes[..., numpy.newaxis] + 1

    # calculate the index of the left side of the zlp by summing values to the left of the zlp position that are above
    # the threshold (1/10). this is done by multiplying the source data by the mask and summing, giving a count. then
    # subtracting that count from the position.
    left_pos_array = zlp_indexes - numpy.sum(src_xdata.data * mask_array > zlp_max_array[...,numpy.newaxis] / 10, axis=-1)

    # calculate the right position too
    right_pos_array = zlp_indexes + (zlp_indexes - left_pos_array) + 1

    # calculate the left and right values using a simple threshold. these two arrays have a 1 where the
    # condition is met; a 0 elsewhere. Multiplied together and then multiplied by the source data results
    # in only values of source data that fall between the two indexes.
    mask_array[:] = (left_pos_array[..., numpy.newaxis] <= indexes[numpy.newaxis, numpy.newaxis, ...]) * (
                indexes[numpy.newaxis, numpy.newaxis, ...] < right_pos_array[..., numpy.newaxis])

    # sum the source data between the two indexes.
    zlp_area_array = numpy.sum(src_xdata * mask_array, axis=-1, dtype=numpy.float32)

    # finally, calculate the thickness as log(total counts / zlp counts).
    thickness_array = numpy.log(numpy.sum(src_xdata.data, axis=-1, dtype=numpy.float32) / zlp_area_array)

    # return the value.
    return DataAndMetadata.new_data_and_metadata(thickness_array, dimensional_calibrations=src_xdata.dimensional_calibrations[:-1])


class EELSThicknessMapping:
    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, spectrum_image_data_item):
        self.__mapped_xdata = map_thickness_xdata(spectrum_image_data_item.xdata)

    def commit(self):
        self.computation.set_referenced_xdata("map", self.__mapped_xdata)


def map_thickness(api, window):
    target_display = window.target_display
    target_data_item_ = target_display._display_item.data_items[0] if target_display and len(target_display._display_item.data_items) > 0 else None
    if target_data_item_ and target_display:
        spectrum_image = Facade.DataItem(target_data_item_)
        map = api.library.create_data_item_from_data(numpy.zeros_like(spectrum_image.display_xdata.data), title="{} Thickness Map".format(spectrum_image.title))
        api.library.create_computation("eels.thickness_mapping", inputs={"spectrum_image_data_item": spectrum_image}, outputs={"map": map})
        window.display_data_item(map)


Symbolic.register_computation_type("eels.thickness_mapping", EELSThicknessMapping)
