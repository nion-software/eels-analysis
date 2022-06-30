# imports
import math
import numpy
import typing

from nion.swift import Facade
from nion.swift.model import Symbolic

DataArrayType = numpy.typing.NDArray[typing.Any]


def sum_zlp(d: DataArrayType) -> typing.Tuple[int, int, int]:
    # Estimates the ZLP, assuming the peak value is the ZLP and that the ZLP is the only gaussian feature in the data.
    # This procedure returns a minimum of three channels for the ZLP integration interval.
    mx_pos = typing.cast(int, numpy.argmax(d))
    mx = d[mx_pos]
    mx_fraction = mx/10
    left_pos = mx_pos - sum(d[:mx_pos + 1] > mx_fraction)
    right_pos = mx_pos + (mx_pos - left_pos) + 1
    s = sum(d[left_pos:right_pos])
    return left_pos, right_pos, s


class MeasureThickness:
    """Carry out the Thickness measurement and add an interval graphic."""

    def __init__(self, computation: Facade.Computation, **kwargs: typing.Any) -> None:
        """Initialize the computation."""
        self.computation = computation

    def execute(self, src: Facade.DataItem, **kwargs: typing.Any) -> None:
        """Execute the computation.

        This method will run in a thread and should not make any modifications to the library.
        """
        xdata = src.display_xdata
        assert xdata
        data = xdata.data
        if data is not None and len(data.shape) == 1:
            self.__data_length = data.shape[0]
            self.__left, self.__right, s = sum_zlp(data)
            self.__thickness  = math.log(sum(data) / s)
            self.__src = src
        else:
            self.__data_length = typing.cast(typing.Any, None)
            self.__left = 0
            self.__right = 0
            self.__thickness = 0
            self.__src = typing.cast(typing.Any, None)

    def commit(self) -> None:
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
            thickness_interval._graphic.color = "#CE00AC"


ComputationCallable = typing.Callable[[Symbolic._APIComputation], Symbolic.ComputationHandlerLike]

def register_measure_thickness_process(api: Facade.API_1) -> None:
    """Registers the measure Thickness computation. This ensures it can be attached and reloaded."""
    api.register_computation_type("nion.eels_analysis.measure_thickness", typing.cast(ComputationCallable, MeasureThickness))


def attach_measure_thickness(api: Facade.API_1, window: Facade.DocumentWindow) -> None:
    """Attaches the measure Thickness computation to the target data item in the window."""
    target_data_item = window.target_data_item
    target_xdata = target_data_item.display_xdata if target_data_item else None
    if target_xdata and target_xdata.is_data_1d:
        api.library.create_computation("nion.eels_analysis.measure_thickness", inputs={"src": target_data_item}, outputs={"thickness_interval": None})
