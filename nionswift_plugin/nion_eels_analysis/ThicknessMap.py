# system imports
import gettext

# third part imports
import numpy

# local libraries
from nion.data import DataAndMetadata
from nion.swift import Facade
from nion.swift.model import Symbolic

_ = gettext.gettext


def sum_zlp(d):
    # estimate the ZLP, assumes the peak value is the ZLP and that the ZLP is the only gaussian feature in the data
    mx_pos = numpy.argmax(d)
    mx = d[mx_pos]
    mx_tenth = mx/10
    left_pos = mx_pos - sum(d[:mx_pos] > mx_tenth)
    right_pos = mx_pos + (mx_pos - left_pos)
    s = sum(d[left_pos:right_pos])
    return left_pos, right_pos, s


class EELSThicknessMapping:
    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, spectrum_image_data_item):
        spectrum_image_xdata = spectrum_image_data_item.xdata
        print(spectrum_image_xdata.data_shape)
        data = numpy.empty((spectrum_image_xdata.data_shape[0:2]))
        for row in range(spectrum_image_xdata.data_shape[0]):
            if row > 0 and row % 10 == 0:
                print(f"Processing row {row} (thickness)")
            for column in range(spectrum_image_xdata.data_shape[1]):
                l, r, s = sum_zlp(spectrum_image_xdata.data[row, column, :])
                data[row, column] = numpy.log(numpy.sum(spectrum_image_xdata.data[row, column, :]) / s)
        dimensional_calibrations = spectrum_image_xdata.dimensional_calibrations[0:-1]
        self.__mapped_xdata = DataAndMetadata.new_data_and_metadata(data, dimensional_calibrations=dimensional_calibrations)

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


class MenuItemDelegate:

    def __init__(self, api):
        self.__api = api
        self.menu_id = "eels_menu"  # required, specify menu_id where this item will go
        self.menu_name = _("EELS")  # optional, specify default name if not a standard menu
        self.menu_before_id = "window_menu"  # optional, specify before menu_id if not a standard menu
        self.menu_item_name = _("Map Thickness")  # menu item name

    def menu_item_execute(self, window):
        map_thickness(self.__api, window)


class MenuExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.eels_analysis.menu_item_map_thickness"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version="~1.0")
        # be sure to keep a reference or it will be closed immediately.
        self.__menu_item_ref = api.create_menu_item(MenuItemDelegate(api))

    def close(self):
        # close will be called when the extension is unloaded. in turn, close any references so they get closed. this
        # is not strictly necessary since the references will be deleted naturally when this object is deleted.
        self.__menu_item_ref.close()
        self.__menu_item_ref = None


Symbolic.register_computation_type("eels.thickness_mapping", EELSThicknessMapping)
