import gettext

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from nion.swift.model import DataItem
from nion.swift.model import Graphics
from nion.swift.model import Symbolic
from nion.data import DataAndMetadata
from nion.data import Calibration
from nion.typeshed import API_1_0 as API

_ = gettext.gettext

kb = 8.617333e-5 # eV/Kelvin


class MeasureTemperature:
    label = _("Measure Temperature")
    inputs = {
        "near_data_item": {"label": _("Near")},
        "far_data_item": {"label": _("Far")},
        "fit_interval_graphic": {"label": _("Fit")},
        }
    outputs = {
        "gain_fit_data_item": {"label": _("Gain fit")},
        "gain_data_item": {"label": _("Gain")},
        "difference_data_item": {"label": _("Difference Near - Far")}
               }

    def __init__(self, computation, **kwargs):
        self.computation = computation
        self.__gain_fit_xdata = None
        self.__gain_xdata = None
        self.__difference_xdata = None
        self.__fit = None

    def execute(self, near_data_item: DataItem.DataItem, far_data_item: DataItem.DataItem, fit_interval_graphic: Graphics.IntervalGraphic):
        try:
            assert near_data_item.xdata.is_data_1d
            assert far_data_item.xdata.is_data_1d
            # Only allow data of same shape for now. A future version could crop to the smaller size of the two.
            assert len(near_data_item.data) == len(far_data_item.data)
            near_xdata = near_data_item.xdata
            far_xdata = far_data_item.xdata
            # For now only allow near and far having the same calibration. A future version could allow different
            # offsets and shift the data accordingly
            assert near_xdata.dimensional_calibrations == far_xdata.dimensional_calibrations

            difference_xdata = near_xdata - far_xdata
            calibration = difference_xdata.dimensional_calibrations[0]
            zero_index = int(calibration.convert_from_calibrated_value(0))
            length = min(zero_index, len(difference_xdata.data) - zero_index)
            loss_slice = slice(zero_index, zero_index + length)
            gain_slice = slice(zero_index-length+1, zero_index+1)
            loss_data = difference_xdata.data[loss_slice]
            gain_data = difference_xdata.data[gain_slice][::-1]
            weights = 1 + np.sqrt(np.abs(near_xdata.data[gain_slice][::-1]) + np.abs(far_xdata[gain_slice][::-1]))

            result_calibration = Calibration.Calibration(offset=calibration.convert_to_calibrated_value(zero_index), scale=calibration.scale, units=calibration.units)
            x_data = result_calibration.convert_to_calibrated_value(np.arange(len(loss_data)))
            interpolator = interp1d(x_data, loss_data)

            def gain_fit(x, T, dx):
                return interpolator(x + 2 * dx) / np.exp((x + dx) / (kb * T))

            fit_slice = slice(int(max(0, fit_interval_graphic.start * len(difference_xdata.data) - zero_index)),
                              int(min(len(gain_data), fit_interval_graphic.end * len(difference_xdata.data) - zero_index)))

            popt, pcov = curve_fit(gain_fit, x_data[fit_slice], gain_data[fit_slice], sigma=weights[fit_slice], p0=(300.0, 0.0))
            self.__fit = popt

            self.__gain_fit_xdata = DataAndMetadata.new_data_and_metadata(gain_fit(x_data[fit_slice], *popt),
                                                                          intensity_calibration=difference_xdata.intensity_calibration,
                                                                          dimensional_calibrations=[result_calibration])
            self.__gain_xdata = DataAndMetadata.new_data_and_metadata(gain_data[fit_slice],
                                                                      intensity_calibration=difference_xdata.intensity_calibration,
                                                                      dimensional_calibrations=[result_calibration])
            self.__difference_xdata = difference_xdata

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(e)
            raise

    def commit(self):
        self.computation.set_referenced_xdata("gain_data_item", self.__gain_xdata)
        self.computation.set_referenced_xdata("gain_fit_data_item", self.__gain_fit_xdata)
        self.computation.set_referenced_xdata("difference_data_item", self.__difference_xdata)
        gain_fit_display_item = self.computation.get_result("gain_fit_data_item").display._display_item
        gain_fit_display_item._set_display_layer_properties(0, label=_(f"Fit T = {self.__fit[0] - 273.15:.0f} °C \nZLP shift = {self.__fit[1]*1000.0:.2f} meV"))



Symbolic.register_computation_type("eels.measure_temperature", MeasureTemperature)


def measure_temperature(api: API.API, window: API.DocumentWindow):
    selected_display_items = window._document_controller._get_two_data_sources()
    document_model = window._document_controller.document_model
    error_msg = "Select two data items each containing one EEL spectrum in order to use this computation."
    assert selected_display_items[0][0] is not None, error_msg
    assert selected_display_items[1][0] is not None, error_msg
    assert selected_display_items[0][0].data_item is not None, error_msg
    assert selected_display_items[1][0].data_item is not None, error_msg
    assert selected_display_items[0][0].data_item.is_data_1d, error_msg
    assert selected_display_items[1][0].data_item.is_data_1d, error_msg

    # First find out which data item is near and which is far. Far should have the higher maximum.
    if np.amax(selected_display_items[0][0].data_item.data) > np.amax(selected_display_items[1][0].data_item.data):
        far_data_item = selected_display_items[0][0].data_item
        near_data_item = selected_display_items[1][0].data_item
    else:
        far_data_item = selected_display_items[1][0].data_item
        near_data_item = selected_display_items[0][0].data_item

    # Now we need to calculate the difference and display it so that we have a place to put the interval on
    difference_xdata = near_data_item.xdata - far_data_item.xdata
    difference_data_item = api.library.create_data_item_from_data_and_metadata(difference_xdata, title=f"Difference (Near - Far), ({near_data_item.title} - {far_data_item.title})")
    window.display_data_item(difference_data_item)
    calibration = difference_xdata.dimensional_calibrations[0]
    # Create the default interval from 20 meV to 100 meV
    graphic = difference_data_item.add_interval_region(calibration.convert_from_calibrated_value(0.02) / len(difference_xdata.data),
                                                       calibration.convert_from_calibrated_value(0.1) / len(difference_xdata.data))

    gain_data_item = api.library.create_data_item(title="Gain")
    gain_fit_data_item = api.library.create_data_item(title="Gain Fit")

    # Create the computation
    api.library.create_computation("eels.measure_temperature",
                                   inputs={
                                       "near_data_item": api._new_api_object(near_data_item),
                                       "far_data_item": api._new_api_object(far_data_item),
                                       "fit_interval_graphic": graphic},
                                   outputs={
                                       "gain_fit_data_item": gain_fit_data_item,
                                       "gain_data_item": gain_data_item,
                                       "difference_data_item": difference_data_item})

    # Set up the plot of Gain and Fit
    window.display_data_item(gain_data_item)
    window.display_data_item(gain_fit_data_item)
    gain_fit_display_item = document_model.get_display_item_for_data_item(gain_fit_data_item._data_item)
    gain_fit_display_item.append_display_data_channel_for_data_item(gain_data_item._data_item)
    gain_fit_display_item._set_display_layer_properties(0, label=_("Fit"), fill_color=None, stroke_color="#F00")
    gain_fit_display_item._set_display_layer_properties(1, label=_("Gain"), fill_color="#1E90FF")
    gain_fit_display_item.set_display_property("legend_position", "top-right")
    gain_fit_display_item.title = "Temperature Measurement Fit"
