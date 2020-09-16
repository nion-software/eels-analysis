from __future__ import annotations

# imports
import copy
import gettext
import numpy
import typing

# local libraries
from nion.data import Calibration
from nion.data import Core
from nion.data import DataAndMetadata
from nion.utils import Registry


_ = gettext.gettext


BackgroundModelParameters = typing.Dict


class AbstractBackgroundModel:
    def __init__(self, background_model_id: str, title: str = None):
        self.background_model_id = background_model_id
        self.title = title
        self.package_title = _("EELS Analysis")

    def fit_background(self, *, spectrum_xdata: DataAndMetadata.DataAndMetadata, fit_intervals: typing.Sequence[Calibration.CalibratedInterval], background_interval: Calibration.CalibratedInterval, **kwargs) -> typing.Dict:
        return {
            "background_model": self.__fit_background(spectrum_xdata, fit_intervals, background_interval),
        }

    def integrate_signal(self, *, spectrum_xdata: DataAndMetadata.DataAndMetadata, fit_intervals: typing.Sequence[Calibration.CalibratedInterval], signal_interval: Calibration.CalibratedInterval, **kwargs) -> typing.Dict:
        # set up initial values
        subtracted_xdata = Core.calibrated_subtract_spectrum(spectrum_xdata, self.__fit_background(spectrum_xdata, fit_intervals, signal_interval))
        if spectrum_xdata.is_navigable:
            return {
                "integrated": DataAndMetadata.new_data_and_metadata(
                    numpy.trapz(subtracted_xdata.data),
                    dimensional_calibrations=spectrum_xdata.navigation_dimensional_calibrations)
            }
        else:
            return {
                "integrated_value": numpy.trapz(subtracted_xdata.data),
            }

    def __fit_background(self, spectrum_xdata: DataAndMetadata.DataAndMetadata, fit_intervals: typing.Sequence[Calibration.CalibratedInterval], background_interval: Calibration.CalibratedInterval) -> DataAndMetadata.DataAndMetadata:
        reference_frame = Calibration.ReferenceFrameAxis(spectrum_xdata.datum_dimensional_calibrations[0], spectrum_xdata.datum_dimension_shape[0])
        # fit polynomial to the data
        xs = numpy.concatenate(
            [Core.get_calibrated_interval_domain(reference_frame, fit_interval) for fit_interval in fit_intervals])
        if len(fit_intervals) > 1:
            ys = numpy.concatenate(
                [Core.get_calibrated_interval_slice(spectrum_xdata, reference_frame, fit_interval).data for fit_interval in
                 fit_intervals])
        else:
            ys = Core.get_calibrated_interval_slice(spectrum_xdata, reference_frame, fit_intervals[0]).data
        # generate background model data from the series
        n = reference_frame.convert_to_pixel(background_interval.end).int_value - reference_frame.convert_to_pixel(
            background_interval.start).int_value
        interval_start = reference_frame.convert_to_calibrated(background_interval.start).value
        interval_end = reference_frame.convert_to_calibrated(background_interval.end).value
        interval_end -= (interval_end - interval_start) / n  # n samples at the left edges of each pixel
        calibration = copy.deepcopy(spectrum_xdata.datum_dimensional_calibrations[0])
        calibration.offset = reference_frame.convert_to_calibrated(background_interval.start).value
        fs = numpy.linspace(interval_start, interval_end, n)
        if spectrum_xdata.is_navigable:
            calibrations = list(copy.deepcopy(spectrum_xdata.navigation_dimensional_calibrations)) + [calibration]
            yss = numpy.reshape(ys, (numpy.product(ys.shape[:-1]),) + (ys.shape[-1],))
            fit_data = self._perform_fits(xs, yss, fs)
            background_xdata = DataAndMetadata.new_data_and_metadata(numpy.reshape(fit_data, ys.shape[:-1] + (n,)),
                                                                     data_descriptor=DataAndMetadata.DataDescriptor(False, spectrum_xdata.navigation_dimension_count, spectrum_xdata.datum_dimension_count),
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
        fit = numpy.empty(yss.shape[:-1] + fs.shape)
        for index in numpy.ndindex(yss.shape[:-1]):
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

    def __init__(self, background_model_id: str, deg: int, transform = None, untransform = None, title: str = None):
        super().__init__(background_model_id, title)
        self.deg = deg
        self.transform = transform
        self.untransform = untransform

    def _perform_fits(self, xs: numpy.ndarray, yss: numpy.ndarray, fs: numpy.ndarray) -> numpy.ndarray:
        transform_data = self.transform or (lambda x: x)
        untransform_data = self.untransform or (lambda x: x)
        fit = untransform_data(numpy.polynomial.polynomial.polyval(fs, numpy.polynomial.polynomial.polyfit(xs, transform_data(yss.transpose()), self.deg)))
        return numpy.where(numpy.isfinite(fit), fit, 0)

    def __unused_perform_fit(self, xs: numpy.ndarray, ys: numpy.ndarray, fs: numpy.ndarray) -> numpy.ndarray:
        # here an an example of using numpy.polynomial.polynomial.Polynomial.fit for when it supports evaluating arrays
        transform_data = self.transform or (lambda x: x)
        untransform_data = self.untransform or (lambda x: x)
        series = typing.cast(typing.Any, numpy.polynomial.polynomial.Polynomial.fit(xs, transform_data(ys), self.deg))
        return untransform_data(series(fs))


class TwoAreaBackgroundModel(AbstractBackgroundModel):
#Fit power law or exponential background model using the two-area method described in Egerton chapter 4.
#This approximation is slightly faster than the polynomial fit for mapping large SI, and may perform better for high-noise spectra.
    def __init__(self, background_model_id: str, params_func = None, model_func = None, title: str = None):
        super().__init__(background_model_id, title)
        self.model_func = model_func
        self.params_func = params_func

    def _perform_fits(self, xs: numpy.ndarray, yss: numpy.ndarray, fs: numpy.ndarray) -> numpy.ndarray:
        if not len(xs) % 2 == 0:
            yss = yss[...,:-1]
            xs = xs[:-1]
        half_interval = len(xs) // 2
        areas_1 = numpy.trapz(yss[...,:half_interval], xs[:half_interval], axis=1)
        areas_2 = numpy.trapz(yss[...,half_interval:], xs[half_interval:], axis=1)
        x_start = xs[0]
        x_center = xs[half_interval]
        x_end = xs[-1]
        p1s, p2s = self.params_func(areas_1, areas_2, x_start, x_center, x_end)
        grid = numpy.transpose(numpy.tile(fs,(len(yss),1)))
        series = typing.cast(typing.Any, numpy.transpose(self.model_func(grid,p1s,p2s)))
        return series


def power_law_params(areas_1: numpy.ndarray, areas_2: numpy.ndarray, x_start: float, x_center: float, x_end: float) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    r = 2 * (numpy.log(areas_1) - numpy.log(areas_2)) / (numpy.log(x_end) - numpy.log(x_start))
    k = 1 - r
    A = k * areas_2 / (x_end ** k - x_center ** k)
    return A, r


def power_law_func(x: numpy.ndarray, A: numpy.ndarray, r: numpy.ndarray) -> numpy.ndarray:
    return A*x**-r
    

# register background models with the registry.
Registry.register_component(TwoAreaBackgroundModel("power_law_two_area_background_model", params_func=power_law_params, model_func=power_law_func, title=_("Power Law Two Area Background")), {"background-model"})
Registry.register_component(PolynomialBackgroundModel("constant_background_model", 0, title=_("Constant Background")), {"background-model"})
Registry.register_component(PolynomialBackgroundModel("linear_background_model", 1, title=_("Linear Background")), {"background-model"})
Registry.register_component(PolynomialBackgroundModel("power_law_background_model", 1, transform=numpy.log, untransform=numpy.exp, title=_("Power Law Background")), {"background-model"})
Registry.register_component(PolynomialBackgroundModel("poly2_background_model", 2, title=_("2nd Order Polynomial Background")), {"background-model"})
Registry.register_component(PolynomialBackgroundModel("poly2_log_background_model", 2, transform=numpy.log, untransform=numpy.exp, title=_("2nd Order Power Law Background")), {"background-model"})
