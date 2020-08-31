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


class PolynomialBackgroundModel:
    def __init__(self, background_model_id: str, deg: int, transform = None, untransform = None, title: str = None):
        self.background_model_id = background_model_id
        self.deg = deg
        self.transform = transform
        self.untransform = untransform
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
        transform_data = self.transform or (lambda x: x)
        # generate background model data from the series
        n = reference_frame.convert_to_pixel(background_interval.end).int_value - reference_frame.convert_to_pixel(
            background_interval.start).int_value
        interval_start = reference_frame.convert_to_calibrated(background_interval.start).value
        interval_end = reference_frame.convert_to_calibrated(background_interval.end).value
        interval_end -= (interval_end - interval_start) / n  # n samples at the left edges of each pixel
        untransform_data = self.untransform or (lambda x: x)
        calibration = copy.deepcopy(spectrum_xdata.datum_dimensional_calibrations[0])
        calibration.offset = reference_frame.convert_to_calibrated(background_interval.start).value
        if spectrum_xdata.is_navigable:
            calibrations = list(copy.deepcopy(spectrum_xdata.navigation_dimensional_calibrations)) + [calibration]
            background_xdata = DataAndMetadata.new_data_and_metadata(numpy.empty(spectrum_xdata.navigation_dimension_shape + (n, )),
                                                                     dimensional_calibrations=calibrations,
                                                                     intensity_calibration=spectrum_xdata.intensity_calibration)
            for index in numpy.ndindex(spectrum_xdata.navigation_dimension_shape):
                background_xdata.data[index] = self.__perform_fit(xs, ys[index], transform_data, untransform_data, n, interval_start, interval_end)
        else:
            poly_data = self.__perform_fit(xs, ys, transform_data, untransform_data, n, interval_start, interval_end)
            background_xdata = DataAndMetadata.new_data_and_metadata(poly_data, dimensional_calibrations=[calibration],
                                                                     intensity_calibration=spectrum_xdata.intensity_calibration)
        return background_xdata

    def __perform_fit(self, xs, ys, transform_data, untransform_data, n, interval_start, interval_end):
        series = typing.cast(typing.Any, numpy.polynomial.polynomial.Polynomial.fit(xs, transform_data(ys), self.deg))
        return untransform_data(series.linspace(n, [interval_start, interval_end])[1])


# register background models with the registry.
Registry.register_component(PolynomialBackgroundModel("constant_background_model", 0, title=_("Constant Background")), {"background-model"})
Registry.register_component(PolynomialBackgroundModel("linear_background_model", 1, title=_("Linear Background")), {"background-model"})
Registry.register_component(PolynomialBackgroundModel("power_law_background_model", 1, transform=numpy.log, untransform=numpy.exp, title=_("Power Law Background")), {"background-model"})
Registry.register_component(PolynomialBackgroundModel("poly2_background_model", 2, title=_("2nd Order Polynomial Background")), {"background-model"})
Registry.register_component(PolynomialBackgroundModel("poly2_log_background_model", 2, transform=numpy.log, untransform=numpy.exp, title=_("2nd Order Power Law Background")), {"background-model"})
