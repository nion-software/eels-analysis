"""
    EELS Analysis Interface for Nion

    A library of functions and classes for general spectral signal extraction techniques,
    such as background subtraction and multiple linear regression of reference signals.

"""

# third party libraries
import numpy

# local libraries
import EELS_EdgeIdentification
import EELS_DataAnalysis
from nion.data import Context
from nion.data import DataAndMetadata

"""
   zero_loss_peak(low_loss_spectra: numpy.ndarray, energy_axis_eV: numpy.ndarray,
                                     energy_origin_eV: float, energy_step_eV: float) -> numpy.ndarray
   edge_signal(edge_spectra: numpy.ndarray, energy_axis_eV: numpy.ndarray,
                                  energy_origin_eV: float, energy_step_eV: float, edge_onset_eV: float, edge_delta_eV: float,
                                  background_ranges_eV: numpy.ndarray) -> numpy.ndarray
   relative_atomic_abundance(edge_spectra: numpy.ndarray, energy_axis_eV: numpy.ndarray,
                                                           energy_origin_eV: float, energy_step_eV: float, 
                                                           atomic_number: int, edge_onset_eV: float, edge_delta_eV: float,
                                                           background_ranges_eV: numpy.ndarray -> numpy.ndarray
   atomic_areal_density_nm2(edge_spectra: numpy.ndarray, edge_energy_axis_eV: numpy.ndarray,
                                                           edge_energy_origin_eV: float, edge_energy_step_eV: float,
                                                           low_loss_spectra: numpy.ndarray, low_loss_axis_eV: numpy.ndarray,
                                                           low_loss_energy_origin_eV: float, low_loss_energy_step_eV: float,
                                                           atomic_number: int, edge_onset_eV: float, edge_delta_eV: float,
                                                           background_ranges_eV: numpy.ndarray -> numpy.ndarray
"""

def extract_signal(data_and_metadata, signal_range, fit_ranges, first_x = 0.0, delta_x = 1.0,
                                                polynomial_order = 1, fit_log_data = False, fit_log_x = False):
    signal_range = numpy.asarray(signal_range) * data_and_metadata.data_shape[0]
    fit_ranges = numpy.asarray(fit_ranges) * data_and_metadata.data_shape[0]
    def data_fn():
        return extract_signal_from_polynomial_background_data(data_and_metadata.data, signal_range, fit_ranges, first_x, delta_x, polynomial_order, fit_log_data, fit_log_x)
    return DataAndMetadata.DataAndMetadata(data_fn, data_and_metadata.data_shape_and_dtype, data_and_metadata.intensity_calibration, data_and_metadata.dimensional_calibrations)


def relative_atomic_abundance(counts_edge: float, partial_cross_section_nm2: float) -> float:
    """Return the relative atomic concentration.

    partial_cross_section is in nm * nm.

    The return value units are atoms / (nm * nm) * spectrum intensity.
    """
    pass


def atomic_areal_density_nm2(counts_edge: float, counts_spectrum: float, partial_cross_section_nm2: float) -> float:
    """Return the areal density.

    partial_cross_section is in nm * nm.

    The return value units are atoms / (nm * nm).
    """
    pass

# register functions
Context.registered_functions["extract_signal_from_polynomial_background"] = extract_signal_from_polynomial_background
Context.registered_functions["subtract_linear_background"] = subtract_linear_background
Context.registered_functions["stacked_subtract_linear_background"] = stacked_subtract_linear_background
