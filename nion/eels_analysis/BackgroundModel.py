from __future__ import annotations

# imports
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

    def analyze_spectrum(self, spectrum: DataAndMetadata.DataAndMetadata, fit_intervals: typing.Sequence[Calibration.CalibratedInterval], **kwargs) -> BackgroundModelParameters:
        # assert that spectrum is 1d with eV calibration
        assert spectrum.is_datum_1d
        assert not spectrum.is_navigable
        assert spectrum.datum_dimensional_calibrations[0].units == "eV"
        # analyze the fit
        reference_frame = Calibration.ReferenceFrameAxis(spectrum.datum_dimensional_calibrations[0], spectrum.datum_dimension_shape[0])
        xs = numpy.concatenate([Core.get_calibrated_interval_domain(reference_frame, fit_interval) for fit_interval in fit_intervals])
        ys = numpy.concatenate([Core.get_calibrated_interval_slice(spectrum, reference_frame, fit_interval).data for fit_interval in fit_intervals])
        transform_data = self.transform or (lambda x: x)
        series = typing.cast(typing.Any, numpy.polynomial.polynomial.Polynomial.fit(xs, transform_data(ys), self.deg))
        # store the fit parameters in the dictionary.
        return {
            "coefficients": numpy.array(series.convert().coef).tolist(),
        }

    def analyze_spectra(self, spectra: DataAndMetadata.DataAndMetadata, fit_intervals: typing.Sequence[Calibration.CalibratedInterval], **kwargs) -> BackgroundModelParameters:
        # assert that spectrum is 1d with eV calibration
        assert spectra.is_datum_1d
        assert spectra.datum_dimensional_calibrations[0].units == "eV"
        # analyze the fit
        reference_frame = Calibration.ReferenceFrameAxis(spectra.datum_dimensional_calibrations[0], spectra.datum_dimension_shape[0])
        xs = numpy.concatenate([Core.get_calibrated_interval_domain(reference_frame, fit_interval) for fit_interval in fit_intervals])
        ys = numpy.concatenate([Core.get_calibrated_interval_slice(spectra, reference_frame, fit_interval).data for fit_interval in fit_intervals], axis=-1)
        transform_data = self.transform or (lambda x: x)
        coefficients = numpy.empty(list(spectra.navigation_dimension_shape) + [self.deg + 1,])
        for index in numpy.ndindex(spectra.navigation_dimension_shape):
            series = typing.cast(typing.Any, numpy.polynomial.polynomial.Polynomial.fit(xs, transform_data(ys[index]), self.deg))
            coefficients[index] = series.convert().coef
        # store the fit parameters in the dictionary.
        return {
            "coefficients": coefficients,
        }

    def generate_background(self, background_model: BackgroundModelParameters, reference_frame: Calibration.ReferenceFrameAxis, interval: Calibration.CalibratedInterval) -> DataAndMetadata:
        n = reference_frame.convert_to_pixel(interval.end).int_value - reference_frame.convert_to_pixel(interval.start).int_value
        interval_start = reference_frame.convert_to_calibrated(interval.start).value
        interval_end = reference_frame.convert_to_calibrated(interval.end).value
        interval_end -= (interval_end - interval_start) / n  # n samples at the left edges of each pixel
        untransform_data = self.untransform or (lambda x: x)
        series = numpy.polynomial.polynomial.Polynomial(background_model["coefficients"])
        poly_data = untransform_data(series.linspace(n, [interval_start, interval_end])[1])
        return DataAndMetadata.new_data_and_metadata(poly_data)

    def generate_backgrounds(self, background_model: BackgroundModelParameters, reference_frame: Calibration.ReferenceFrameAxis, interval: Calibration.CalibratedInterval) -> DataAndMetadata:
        coefficients = background_model["coefficients"]
        n = reference_frame.convert_to_pixel(interval.end).int_value - reference_frame.convert_to_pixel(interval.start).int_value
        interval_start = reference_frame.convert_to_calibrated(interval.start).value
        interval_end = reference_frame.convert_to_calibrated(interval.end).value
        interval_end -= (interval_end - interval_start) / n  # n samples at the left edges of each pixel
        untransform_data = self.untransform or (lambda x: x)
        mapped_data = numpy.empty(list(coefficients.shape[:-1]) + [n,])
        for index in numpy.ndindex(mapped_data.shape[:-1]):
            series = numpy.polynomial.polynomial.Polynomial(coefficients[index])
            mapped_data[index] = untransform_data(series.linspace(n, [interval_start, interval_end])[1])
        return DataAndMetadata.new_data_and_metadata(mapped_data)


# register background models with the registry.
Registry.register_component(PolynomialBackgroundModel("constant_background_model", 0, title=_("Constant Background")), {"background-model"})
Registry.register_component(PolynomialBackgroundModel("linear_background_model", 1, title=_("Linear Background")), {"background-model"})
Registry.register_component(PolynomialBackgroundModel("power_law_background_model", 1, transform=numpy.log, untransform=numpy.exp, title=_("Power Law Background")), {"background-model"})
Registry.register_component(PolynomialBackgroundModel("poly2_background_model", 2, title=_("2nd Order Polynomial Background")), {"background-model"})
Registry.register_component(PolynomialBackgroundModel("poly2_log_background_model", 2, transform=numpy.log, untransform=numpy.exp, title=_("2nd Order Power Law Background")), {"background-model"})
