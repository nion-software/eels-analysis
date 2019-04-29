# imports
import math

# local libraries
# None


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
        data = src.display_xdata.data
        if data is not None and len(data.shape) == 1:
            self.__data_length = data.shape[0]
            self.__left, self.__right, s = sum_zlp(data)
            self.__thickness  = math.log(sum(data) / s)
            self.__src = src
        else:
            self.__data_length = None
            self.__left = 0
            self.__right = 0
            self.__thickness = 0
            self.__src = None

    def commit(self):
        """Commit the computation.

        This method will run at UI time and can make modifications to the library. It is essential
        that this method be as fast as possible. Any lengthy operations should be done in `execute`.
        """
        if self.__src:
            left, right, thickness = self.__left, self.__right, self.__thickness
            data_length = self.__data_length
            start = left / data_length
            end = right / data_length
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
    if target_data_item and target_data_item.display_xdata.is_data_1d:
        api.library.create_computation("nion.eels_analysis.measure_thickness", inputs={"src": target_data_item}, outputs={"thickness_interval": None})
