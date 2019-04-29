# imports
# None

# local libraries
from nion.eels_analysis import ZLP_Analysis


class MeasureZLP:
    """Carry out the ZLP measurement and add an interval graphic."""

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
            self.__amplitude, self.__pos, self.__left, self.__right = ZLP_Analysis.estimate_zlp_amplitude_position_width_com(data)
            self.__src = src
        else:
            self.__data_length = None
            self.__amplitude = 0
            self.__src = None

    def commit(self):
        """Commit the computation.

        This method will run at UI time and can make modifications to the library. It is essential
        that this method be as fast as possible. Any lengthy operations should be done in `execute`.
        """
        if self.__src:
            amplitude, pos, left, right = self.__amplitude, self.__pos, self.__left, self.__right
            data_length = self.__data_length
            start = left / data_length
            end = right / data_length
            zlp_interval = self.computation.get_result("zlp_interval", None)
            if not zlp_interval:
                zlp_interval = self.__src.add_interval_region(start, end)
                self.computation.set_result("zlp_interval", zlp_interval)
            zlp_interval.interval = start, end
            zlp_interval.graphic_id = "zlp_interval"
            zlp_interval._graphic.color = "#0F0"


def register_measure_zlp_process(api):
    """Registers the measure ZLP computation. This ensures it can be attached and reloaded."""
    api.register_computation_type("nion.eels_analysis.measure_zlp", MeasureZLP)


def attach_measure_zlp(api, window):
    """Attaches the measure ZLP computation to the target data item in the window."""
    target_data_item = window.target_data_item
    if target_data_item and target_data_item.display_xdata.is_data_1d:
        api.library.create_computation("nion.eels_analysis.measure_zlp", inputs={"src": target_data_item}, outputs={"zlp_interval": None})
