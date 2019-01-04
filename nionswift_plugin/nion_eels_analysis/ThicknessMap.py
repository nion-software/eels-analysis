# imports
import numpy
import typing

# local libraries
from nion.data import DataAndMetadata
from nion.swift import Facade
from nion.swift.model import Symbolic


def sum_zlp(d):
    # estimate the ZLP, assumes the peak value is the ZLP and that the ZLP is the only gaussian feature in the data
    mx_pos = numpy.argmax(d)
    mx = d[mx_pos]
    mx_tenth = mx/10
    left_pos = mx_pos - sum(d[:mx_pos] > mx_tenth)
    right_pos = mx_pos + (mx_pos - left_pos)
    s = sum(d[left_pos:right_pos])
    return left_pos, right_pos, s


def map_thickness_xdata(src_xdata: DataAndMetadata.DataAndMetadata, progress_fn=None) -> typing.Optional[DataAndMetadata.DataAndMetadata]:
    data = numpy.empty((src_xdata.data_shape[0:2]), numpy.float32)
    for row in range(src_xdata.data_shape[0]):
        if row > 0 and row % 10 == 0:
            if callable(progress_fn):
                progress_fn(row)
        for column in range(src_xdata.data_shape[1]):
            l, r, s = sum_zlp(src_xdata.data[row, column, :])
            data[row, column] = numpy.log(numpy.sum(src_xdata.data[row, column, :]) / s)
    dimensional_calibrations = src_xdata.dimensional_calibrations[0:-1]
    return DataAndMetadata.new_data_and_metadata(data, dimensional_calibrations=dimensional_calibrations)


class EELSThicknessMapping:
    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, spectrum_image_data_item):
        def progress(row):
            print(f"Processing row {row} (thickness)")

        self.__mapped_xdata = map_thickness_xdata(spectrum_image_data_item.xdata, progress)

    def commit(self):
        self.computation.set_referenced_xdata("map", self.__mapped_xdata)


def map_thickness(api, window):
    target_display = window.target_display
    target_data_item_ = target_display._display_item.data_items[0] if target_display and len(target_display._display_item.data_items) > 0 else None
    if target_data_item_ and target_display:
        spectrum_image = Facade.DataItem(target_data_item_)
        map = api.library.create_data_item_from_data(numpy.zeros_like(spectrum_image.display_xdata.data), title="{} Thickness Map".format(spectrum_image.title))
        computation = api.library.create_computation("eels.thickness_mapping", inputs={"spectrum_image_data_item": spectrum_image}, outputs={"map": map})
        computation._computation.source = spectrum_image._data_item
        window.display_data_item(map)


Symbolic.register_computation_type("eels.thickness_mapping", EELSThicknessMapping)
