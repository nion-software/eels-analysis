# system imports
import gettext
import math

# third part imports
import numpy

# local libraries
# None

_ = gettext.gettext


import numpy


def sum_zlp(d):
    # estimate the ZLP, assumes the peak value is the ZLP and that the ZLP is the only gaussian feature in the data
    mx_pos = numpy.argmax(d)
    mx = d[mx_pos]
    mx_tenth = mx/10
    left_pos = mx_pos - sum(d[:mx_pos] > mx_tenth)
    right_pos = mx_pos + (mx_pos - left_pos)
    s = sum(d[left_pos:right_pos])
    return left_pos, right_pos, s


class MeasureThickness:
    """Carry out the Thickness measurement and add an interval graphic."""

    def __init__(self, computation, **kwargs):
        """Initialize the computation."""
        self.computation = computation

    def execute(self, src):
        """Execute the computation.

        This method will run in a thread and should not make any modifications to the library.
        """
        self.__data = src.display_xdata.data
        self.__left, self.__right, s = sum_zlp(self.__data)
        self.__thickness  = math.log(sum(self.__data) / s)
        self.__src = src

    def commit(self):
        """Commit the computation.

        This method will run at UI time and can make modifications to the library. It is essential
        that this method be as fast as possible. Any lengthy operations should be done in `execute`.
        """
        left, right, thickness = self.__left, self.__right, self.__thickness
        data = self.__data
        start = left / data.shape[-1]
        end = right / data.shape[-1]
        thickness_interval = self.computation.get_result("thickness_interval", None)
        if not thickness_interval:
            thickness_interval = self.__src.add_interval_region(start, end)
            self.computation.set_result("thickness_interval", thickness_interval)
        thickness_interval.interval = start, end
        thickness_interval.graphic_id = "thickness_interval"
        thickness_interval.label = f"{self.__thickness:0.4f}"
        thickness_interval._graphic.color = "#0F0"


def register_measure_thickness_process(api):
    """Registers the measure Thickness computation. This ensures it can be attached and reloaded."""
    api.register_computation_type("nion.eels_analysis.measure_thickness", MeasureThickness)


def attach_measure_thickness(api, window):
    """Attaches the measure Thickness computation to the target data item in the window."""
    target_data_item = window.target_data_item
    if target_data_item:
        api.library.create_computation("nion.eels_analysis.measure_thickness", inputs={"src": target_data_item}, outputs={"thickness_interval": None})


class MenuItemDelegate(object):

    def __init__(self, api):
        self.__api = api
        self.menu_id = "eels_menu"  # required, specify menu_id where this item will go
        self.menu_name = _("EELS")  # optional, specify default name if not a standard menu
        self.menu_before_id = "window_menu"  # optional, specify before menu_id if not a standard menu
        self.menu_item_name = _("Show Live Thickness Measurement")  # menu item name

    def menu_item_execute(self, window):
        attach_measure_thickness(self.__api, window)


class MenuExampleExtension(object):

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.eels_analysis.menu_item_attach_live_thickness"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version="~1.0")
        # be sure to keep a reference or it will be closed immediately.
        self.__menu_item_ref = api.create_menu_item(MenuItemDelegate(api))
        register_measure_thickness_process(api)

    def close(self):
        # close will be called when the extension is unloaded. in turn, close any references so they get closed. this
        # is not strictly necessary since the references will be deleted naturally when this object is deleted.
        self.__menu_item_ref.close()
        self.__menu_item_ref = None
