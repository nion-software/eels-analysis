# imports
import typing

# local libraries
from nion.eels_analysis import ZLP_Analysis
from nion.swift import Facade
from nion.swift.model import DisplayItem
from nion.swift.model import Symbolic


class MeasureZLP:
    """Carry out the ZLP measurement and add an interval graphic."""

    def __init__(self, computation: Facade.Computation, **kwargs: typing.Any) -> None:
        """Initialize the computation."""
        self.computation = computation

    def execute(self, src: Facade.DataItem, **kwargs: typing.Any) -> None:
        """Execute the computation.

        This method will run in a thread and should not make any modifications to the library.
        """
        assert src.xdata
        data = src.xdata.data
        if data is not None and len(data.shape) == 1:
            self.__data_length = data.shape[0]
            self.__amplitude, self.__pos, self.__left, self.__right = ZLP_Analysis.estimate_zlp_amplitude_position_width_com(data)
        else:
            self.__data_length = typing.cast(typing.Any, None)
            self.__amplitude = 0

    def commit(self) -> None:
        """Commit the computation.

        This method will run at UI time and can make modifications to the library. It is essential
        that this method be as fast as possible. Any lengthy operations should be done in `execute`.
        """
        if self.__data_length is not None:
            amplitude, pos, left, right = self.__amplitude, self.__pos, self.__left, self.__right
            data_length = self.__data_length
            start = left / data_length
            end = right / data_length
            zlp_interval = self.computation.get_result("zlp_interval", None)
            if not zlp_interval:
                src = self.computation.get_input("src")
                zlp_interval = src.add_interval_region(start, end)
                self.computation.set_result("zlp_interval", zlp_interval)
            zlp_interval.interval = start, end
            zlp_interval.graphic_id = "zlp_interval"
            zlp_interval._graphic.color = "black"
            zlp_interval._graphic.role = "measurement"

ComputationCallable = typing.Callable[[Symbolic._APIComputation], Symbolic.ComputationHandlerLike]

def register_measure_zlp_process(api: Facade.API_1) -> None:
    """Registers the measure ZLP computation. This ensures it can be attached and reloaded."""
    api.register_computation_type("nion.eels_analysis.measure_zlp", typing.cast(ComputationCallable, MeasureZLP))


def can_measure_zlp(display_item: DisplayItem.DisplayItem | None) -> bool:
    """Return whether the live ZLP measurement can be attached to the display item's data item.

    It needs the display item to show a spectrum, such as a spectrum or one of a sequence of spectra. The display data
    shape comes from the metadata, so the data isn't loaded.
    """
    data_item = display_item.data_item if display_item else None
    display_data_channel = display_item.display_data_channel if display_item else None
    display_data_shape = display_data_channel.display_data_shape if display_data_channel else None
    return data_item is not None and display_data_shape is not None and len(display_data_shape) == 1


def attach_measure_zlp(api: Facade.API_1, window: Facade.DocumentWindow) -> None:
    """Attaches the measure ZLP computation to the target data item in the window."""
    target_display = window.target_display
    target_data_item = window.target_data_item
    if target_display and target_data_item and can_measure_zlp(target_display._display_item):
        api.library.create_computation("nion.eels_analysis.measure_zlp", inputs={"src": target_data_item}, outputs={"zlp_interval": None})
