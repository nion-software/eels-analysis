from __future__ import annotations

# imports
import copy
import gettext
import numpy
import typing

# local libraries
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.utils import Registry


_ = gettext.gettext


class AbstractZeroLossPeakModel:
    def __init__(self, zero_loss_peak_model_id: str, title: str = None):
        self.zero_loss_peak_model_id = zero_loss_peak_model_id
        self.title = title
        self.package_title = _("EELS Analysis")

    def fit_zero_loss_peak(self, *, spectrum_xdata: DataAndMetadata.DataAndMetadata, **kwargs) -> typing.Dict:
        return {
            "zero_loss_peak_model": self.__fit_zero_loss_peak(spectrum_xdata),
        }

    def __fit_zero_loss_peak(self, spectrum_xdata: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.DataAndMetadata:
        z = int(spectrum_xdata.dimensional_calibrations[-1].convert_from_calibrated_value(0.0))
        calibration = copy.deepcopy(spectrum_xdata.datum_dimensional_calibrations[0])
        ys = spectrum_xdata.data
        if spectrum_xdata.is_navigable:
            calibrations = list(copy.deepcopy(spectrum_xdata.navigation_dimensional_calibrations)) + [calibration]
            yss = numpy.reshape(ys, (numpy.product(ys.shape[:-1]),) + (ys.shape[-1],))
            fit_data = self._perform_fits(yss, z)
            data_descriptor = DataAndMetadata.DataDescriptor(False, spectrum_xdata.navigation_dimension_count,
                                                             spectrum_xdata.datum_dimension_count)
            model_xdata = DataAndMetadata.new_data_and_metadata(
                numpy.reshape(fit_data, ys.shape[:-1] + (ys.shape[-1],)),
                data_descriptor=data_descriptor,
                dimensional_calibrations=calibrations,
                intensity_calibration=spectrum_xdata.intensity_calibration)
        else:
            poly_data = self._perform_fit(ys, z)
            model_xdata = DataAndMetadata.new_data_and_metadata(poly_data, dimensional_calibrations=[calibration],
                                                                intensity_calibration=spectrum_xdata.intensity_calibration)
        return model_xdata

    def _perform_fits(self, yss: numpy.ndarray, z: int) -> numpy.ndarray:
        # ys will be an array of y-values with shape (m,L)
        # z is the index of the column of 0eV
        # return an ndarray of the fit with shape (m,L)
        # implement at least one of _perform_fits and _perform_fit
        fit = numpy.empty(yss.shape)
        for index in numpy.ndindex(yss.shape[:-1]):
            fit[index] = self._perform_fit(yss[index], z)
        return fit

    def _perform_fit(self, ys: numpy.ndarray, z: int) -> numpy.ndarray:
        # ys will be an array of y-values with shape (L)
        # z is the index of the column of 0eV
        # return an ndarray of the fit with shape (L)
        # implement at least one of _perform_fits and _perform_fit
        return numpy.reshape(self._perform_fits(numpy.reshape(ys, (1,) + ys.shape), z), ys.shape)


class SimpleZeroLossPeakModel(AbstractZeroLossPeakModel):

    def __init__(self, zero_loss_peak_model: str, title: str = None):
        super().__init__(zero_loss_peak_model, title)

    def _perform_fits(self, yss: numpy.ndarray, z: int) -> numpy.ndarray:
        left = max(0, z - 3)
        right = min(yss.shape[-1], z + 3)
        result = numpy.zeros(yss.shape)
        result[..., left:right] = yss[..., left:right]
        # print(f"{z=} {left=} {right=}")
        return result


# register models with the registry.
Registry.register_component(SimpleZeroLossPeakModel("simple_peak_model", title=_("Simple")), {"zlp-model"})
