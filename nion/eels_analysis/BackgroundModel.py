from __future__ import annotations

# imports
import copy
import gettext
import numpy
import typing

# local libraries
from nion.data import Core
from nion.data import DataAndMetadata
from nion.utils import Registry


_ = gettext.gettext


BackgroundModelParameters = typing.Dict
BackgroundInterval = typing.Tuple[float, float]


def get_calibrated_interval_slice(spectrum: DataAndMetadata.DataAndMetadata,
                                  interval: BackgroundInterval) -> DataAndMetadata.DataAndMetadata:
    assert spectrum.is_datum_1d
    start_px = round(spectrum.data_shape[-1] * interval[0])
    stop_px = round(spectrum.data_shape[-1] * interval[1])
    return spectrum[..., start_px:stop_px]


def get_calibrated_interval_domain(spectrum: DataAndMetadata.DataAndMetadata,
                                   interval: BackgroundInterval) -> DataAndMetadata.DataAndMetadata:
    calibration = spectrum.dimensional_calibrations[-1]
    start = calibration.convert_to_calibrated_value(interval[0] * spectrum.data_shape[-1])
    end = calibration.convert_to_calibrated_value(interval[1] * spectrum.data_shape[-1])
    start_px = round(spectrum.data_shape[-1] * interval[0])
    stop_px = round(spectrum.data_shape[-1] * interval[1])
    return DataAndMetadata.new_data_and_metadata(
        numpy.linspace(start, end, (stop_px - start_px), endpoint=False, dtype=numpy.float32),
        dimensional_calibrations=[calibration])


class AbstractBackgroundModel:
    def __init__(self, background_model_id: str, title: str = None):
        self.background_model_id = background_model_id
        self.title = title
        self.package_title = _("EELS Analysis")

    def fit_background(self, *, spectrum_xdata: DataAndMetadata.DataAndMetadata,
                       fit_intervals: typing.Sequence[BackgroundInterval],
                       background_interval: BackgroundInterval, **kwargs) -> typing.Dict:
        return {
            "background_model": self.__fit_background(spectrum_xdata, fit_intervals, background_interval),
        }

    def subtract_background(self, *, spectrum_xdata: DataAndMetadata.DataAndMetadata,
                            fit_intervals: typing.Sequence[BackgroundInterval], **kwargs) -> typing.Dict:
        # set up initial values
        fit_minimum = min([fit_interval[0] for fit_interval in fit_intervals])
        signal_interval = fit_minimum, 1.0
        subtracted_xdata = Core.calibrated_subtract_spectrum(spectrum_xdata, self.__fit_background(spectrum_xdata, fit_intervals, signal_interval))
        assert subtracted_xdata
        return {"subtracted": subtracted_xdata}

    def integrate_signal(self, *, spectrum_xdata: DataAndMetadata.DataAndMetadata,
                         fit_intervals: typing.Sequence[BackgroundInterval],
                         signal_interval: BackgroundInterval, **kwargs) -> typing.Dict:
        # set up initial values
        subtracted_xdata = Core.calibrated_subtract_spectrum(spectrum_xdata, self.__fit_background(spectrum_xdata, fit_intervals, signal_interval))
        assert subtracted_xdata
        subtracted_data = subtracted_xdata.data
        assert subtracted_data is not None
        if spectrum_xdata.is_navigable:
            return {
                "integrated": DataAndMetadata.new_data_and_metadata(
                    numpy.trapz(subtracted_data),
                    dimensional_calibrations=spectrum_xdata.navigation_dimensional_calibrations)
            }
        else:
            return {
                "integrated_value": numpy.trapz(subtracted_data),
            }

    def __fit_background(self, spectrum_xdata: DataAndMetadata.DataAndMetadata,
                         fit_intervals: typing.Sequence[BackgroundInterval],
                         background_interval: BackgroundInterval) -> DataAndMetadata.DataAndMetadata:
        # fit polynomial to the data
        xs = numpy.concatenate(
            [get_calibrated_interval_domain(spectrum_xdata, fit_interval) for fit_interval in fit_intervals],
            dtype=numpy.float32)
        ys: numpy.typing.NDArray[typing.Any]
        if len(fit_intervals) > 1:
            ys = numpy.concatenate(
                [get_calibrated_interval_slice(spectrum_xdata, fit_interval)._data_ex for fit_interval in
                 fit_intervals])
        else:
            ys = get_calibrated_interval_slice(spectrum_xdata, fit_intervals[0])._data_ex
        # generate background model data from the series
        background_interval_start_pixel = round(spectrum_xdata.data_shape[-1] * background_interval[0])
        background_interval_end_pixel = round(spectrum_xdata.data_shape[-1] * background_interval[1])
        n = background_interval_end_pixel - background_interval_start_pixel
        calibration = copy.deepcopy(spectrum_xdata.dimensional_calibrations[-1])
        interval_start = calibration.convert_to_calibrated_value(background_interval_start_pixel)
        interval_end = calibration.convert_to_calibrated_value(background_interval_end_pixel)
        interval_end -= (interval_end - interval_start) / n  # n samples at the left edges of each pixel
        calibration.offset = interval_start
        fs = numpy.linspace(interval_start, interval_end, n, dtype=numpy.float32)
        if spectrum_xdata.is_navigable:
            calibrations = list(copy.deepcopy(spectrum_xdata.navigation_dimensional_calibrations)) + [calibration]
            yss = numpy.reshape(ys, (numpy.product(ys.shape[:-1]),) + (ys.shape[-1],))
            fit_data = self._perform_fits(xs, yss, fs)
            data_descriptor = DataAndMetadata.DataDescriptor(False, spectrum_xdata.navigation_dimension_count,
                                                             spectrum_xdata.datum_dimension_count)
            background_xdata = DataAndMetadata.new_data_and_metadata(numpy.reshape(fit_data, ys.shape[:-1] + (n,)),
                                                                     data_descriptor=data_descriptor,
                                                                     dimensional_calibrations=calibrations,
                                                                     intensity_calibration=spectrum_xdata.intensity_calibration)
        else:
            poly_data = self._perform_fit(xs, ys, fs)
            background_xdata = DataAndMetadata.new_data_and_metadata(poly_data, dimensional_calibrations=[calibration],
                                                                     intensity_calibration=spectrum_xdata.intensity_calibration)
        return background_xdata

    def _perform_fits(self, xs: numpy.ndarray, yss: numpy.ndarray, fs: numpy.ndarray) -> numpy.ndarray:
        # xs will be a set of x-values with shape (L) representing the energies at which to fit
        # ys will be an array of y-values with shape (m,L)
        # fs will be an array of x-values with shape (n) representing energies at which to generate fitted data
        # return an ndarray of the fit with shape (m,n)
        # implement at least one of _perform_fits and _perform_fit
        yss_shape = yss.shape[:-1]
        fit = numpy.empty(yss_shape + fs.shape)
        for index in numpy.ndindex(*yss_shape):
            fit[index] = self._perform_fit(xs, yss[index], fs)
        return fit

    def _perform_fit(self, xs: numpy.ndarray, ys: numpy.ndarray, fs: numpy.ndarray) -> numpy.ndarray:
        # xs will be a set of x-values with shape (L) representing the energies at which to fit
        # ys will be an array of y-values with shape (L)
        # fs will be an array of x-values with shape (n) representing energies at which to generate fitted data
        # return an ndarray of the fit with shape (n)
        # implement at least one of _perform_fits and _perform_fit
        return numpy.reshape(self._perform_fits(xs, numpy.reshape(ys, (1,) + ys.shape), fs), fs.shape)


class PolynomialBackgroundModel(AbstractBackgroundModel):

    def __init__(self, background_model_id: str, deg: int, transform=None, untransform=None, title: str = None):
        super().__init__(background_model_id, title)
        self.deg = deg
        self.transform = transform
        self.untransform = untransform

    def _perform_fits(self, xs: numpy.ndarray, yss: numpy.ndarray, fs: numpy.ndarray) -> numpy.ndarray:
        transform_data = self.transform or (lambda x: x)
        untransform_data = self.untransform or (lambda x: x)
        coefficients = numpy.polynomial.polynomial.polyfit(transform_data(xs), transform_data(yss.transpose()), self.deg)
        fit = untransform_data(numpy.polynomial.polynomial.polyval(transform_data(fs), coefficients))
        return numpy.where(numpy.isfinite(fit), fit, 0)

    def __unused_perform_fit(self, xs: numpy.ndarray, ys: numpy.ndarray, fs: numpy.ndarray) -> numpy.ndarray:
        # here an an example of using numpy.polynomial.polynomial.Polynomial.fit for when it supports evaluating arrays
        transform_data = self.transform or (lambda x: x)
        untransform_data = self.untransform or (lambda x: x)
        series = typing.cast(typing.Any, numpy.polynomial.polynomial.Polynomial.fit(xs, transform_data(ys), self.deg))
        return untransform_data(series(fs))


class TwoAreaBackgroundModel(AbstractBackgroundModel):
    # Fit power law or exponential background model using the two-area method described in Egerton chapter 4.
    # This approximation is slightly faster than the polynomial fit for mapping large SI, and may perform better for high-noise spectra.

    def __init__(self,
                 background_model_id: str,
                 params_func: typing.Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, float, float, float],
                                              typing.Tuple[numpy.ndarray, numpy.ndarray]],
                 model_func: typing.Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray], numpy.ndarray],
                 title: str = None):
        super().__init__(background_model_id, title)
        self.model_func = model_func
        self.params_func = params_func

    def _perform_fits(self, xs: numpy.ndarray, yss: numpy.ndarray, fs: numpy.ndarray) -> numpy.ndarray:
        half_interval = len(xs) // 2
        x_interval_1 = xs[:half_interval]
        x_interval_2 = xs[half_interval:2 * half_interval]
        y_interval_1 = yss[..., :half_interval]
        y_interval_2 = yss[..., half_interval:2 * half_interval]
        x_start = xs[0]
        x_center = xs[half_interval]
        x_end = xs[-1]
        params = self.params_func(x_interval_1, x_interval_2, y_interval_1, y_interval_2, x_start, x_center, x_end)
        xs_tiled = numpy.transpose(numpy.tile(fs, (len(yss), 1)))
        series = typing.cast(typing.Any, numpy.transpose(self.model_func(xs_tiled, *params)))
        return series


def power_law_params(x_interval_1: numpy.ndarray,
                     x_interval_2: numpy.ndarray,
                     y_interval_1: numpy.ndarray,
                     y_interval_2: numpy.ndarray,
                     x_start: float,
                     x_center: float,
                     x_end: float) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    areas_1 = typing.cast(numpy.ndarray, numpy.trapz(y_interval_1, x_interval_1, axis=1))
    areas_2 = typing.cast(numpy.ndarray, numpy.trapz(y_interval_2, x_interval_2, axis=1))
    r = 2 * (numpy.log(areas_1) - numpy.log(areas_2)) / (numpy.log(x_end) - numpy.log(x_start))
    k = 1 - r
    A = k * areas_2 / (x_end ** k - x_center ** k)
    return A, r


def power_law_func(x: numpy.ndarray, A: numpy.ndarray, r: numpy.ndarray) -> numpy.ndarray:
    return A * x ** -r


def exponential_params(x_interval_1: numpy.ndarray,
                       x_interval_2: numpy.ndarray,
                       y_interval_1: numpy.ndarray,
                       y_interval_2: numpy.ndarray,
                       x_start: float,
                       x_center: float,
                       x_end: float) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    y_log_1 = numpy.log(y_interval_1)
    y_log_2 = numpy.log(y_interval_2)
    geo_mean_1 = numpy.exp(numpy.mean(y_log_1))
    geo_mean_2 = numpy.exp(numpy.mean(y_log_2))
    x1 = (x_start + x_center) / 2
    x2 = (x_center + x_end) / 2
    A = numpy.exp((numpy.log(geo_mean_1) - (x1 / x2) * numpy.log(geo_mean_2)) / (1 - x1 / x2))
    tau = -x2 / (numpy.log(geo_mean_2) - numpy.log(A))
    return A, tau


def exponential_func(x: numpy.ndarray, A: numpy.ndarray, tau: numpy.ndarray) -> numpy.ndarray:
    return A * numpy.exp(-x / tau)


# register background models with the registry.
Registry.register_component(PolynomialBackgroundModel("constant_background_model", 0,
                                                      title=_("Constant")), {"background-model"})

Registry.register_component(PolynomialBackgroundModel("linear_background_model", 1,
                                                      title=_("Linear")), {"background-model"})

Registry.register_component(PolynomialBackgroundModel("power_law_background_model", 1,
                                                      transform=numpy.log, untransform=numpy.exp, title=_("Power Law")), {"background-model"})

Registry.register_component(PolynomialBackgroundModel("poly2_background_model", 2,
                                                      title=_("2nd Order Polynomial")), {"background-model"})

Registry.register_component(PolynomialBackgroundModel("poly2_log_background_model", 2, transform=numpy.log, untransform=numpy.exp,
                                                      title=_("2nd Order Power Law")), {"background-model"})

Registry.register_component(TwoAreaBackgroundModel("power_law_two_area_background_model", params_func=power_law_params, model_func=power_law_func,
                                                   title=_("Power Law Two Area")), {"background-model"})

Registry.register_component(TwoAreaBackgroundModel("exponential_two_area_background_model", params_func=exponential_params, model_func=exponential_func,
                                                   title=_("Exponential Two Area")), {"background-model"})
