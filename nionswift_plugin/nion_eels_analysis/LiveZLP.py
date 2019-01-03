# imports
import math
import numpy
import scipy.interpolate
import scipy.optimize
import scipy.signal
import scipy.stats

# local libraries
# None


def estimate_zlp_amplitude_position_width(d):
    # estimate the ZLP, assumes the peak value is the ZLP and that the ZLP is the only gaussian feature in the data
    gaussian = lambda x, a, b, c: a*numpy.exp(-(x-b)**2/(2*c**2))
    d_max = numpy.amax(d)
    # first fit a bspline to the data
    s = scipy.interpolate.splrep(range(d.shape[-1]), d - d_max / 2)
    # assuming bspline has two roots, use them to estimate FWHM
    r = scipy.interpolate.sproot(s)
    if len(r) == 2:
        fwhm = r[1] - r[0]
        c = fwhm / (2 * math.sqrt(2 * math.log(2)))
        # now fit the gaussian to the data, using the amplitude, std dev, and bspline position as estimates (10%)
        popt, pcov = scipy.optimize.curve_fit(gaussian, range(d.shape[0]), d, bounds=([d_max * 0.9, r[0], c * 0.9], [d_max * 1.1, r[1], c * 1.1]))
        return popt
    return numpy.nan, numpy.nan, numpy.nan


def estimate_zlp_amplitude_position_width_counting(d):
    # estimate the ZLP, assumes the peak value is the ZLP and that the ZLP is the only gaussian feature in the data
    mx_pos = numpy.argmax(d)
    mx = d[mx_pos]
    half_mx = mx/2
    left_pos = mx_pos - sum(d[:mx_pos] > half_mx)
    right_pos = mx_pos + sum(d[mx_pos:] > half_mx)
    return mx, mx_pos, left_pos, right_pos


class MeasureZLP:
    """Carry out the ZLP measurement and add an interval graphic."""

    def __init__(self, computation, **kwargs):
        """Initialize the computation."""
        self.computation = computation

    def execute(self, src):
        """Execute the computation.

        This method will run in a thread and should not make any modifications to the library.
        """
        self.__data = src.display_xdata.data
        self.__amplitude, self.__pos, self.__left, self.__right = estimate_zlp_amplitude_position_width_counting(self.__data)
        self.__src = src

    def commit(self):
        """Commit the computation.

        This method will run at UI time and can make modifications to the library. It is essential
        that this method be as fast as possible. Any lengthy operations should be done in `execute`.
        """
        amplitude, pos, left, right = self.__amplitude, self.__pos, self.__left, self.__right
        data = self.__data
        start = left / data.shape[-1]
        end = right / data.shape[-1]
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
    if target_data_item:
        api.library.create_computation("nion.eels_analysis.measure_zlp", inputs={"src": target_data_item}, outputs={"zlp_interval": None})
