"""
    Spectral Analysis Library

    A library of functions and classes for general spectral signal extraction techniques,
    such as background subtraction and multiple linear regression of reference signals.

"""

# standard libraries
# None

# third party libraries
import numpy

# local libraries
from . import CurveFitting
from nion.data import Context
from nion.data import DataAndMetadata


def extract_signal_from_polynomial_background_data(data, signal_range, fit_ranges, first_x = 0.0, delta_x = 1.0,
                                                polynomial_order = 1, fit_log_data = False, fit_log_x = False):

    """A function for performing generic polynomial background subtraction on 1-D spectral data arrays.

    The required data (NumPy) array can have 1 or 2 dimensions.  If it is 1-dimensional, then each array element is
    a spectral intensity (ordinate) value along an equi-sampled x axis with an initial abcissa value given by first_x and
    a fixed abcissa increment per data element given by delta_x.  If the data array is 2-dimensional, then one of the
    dimensions must have size 2.  The first entry along that dimension is an x (abcissa) value and the second entry is
    the corresponding spectral intensity (ordinate) value.  Note that in this case, the abcissa values need not be ordered.
    This function will return background-subtracted spectral intensities regardless of the abcissa value ordering, or lack thereof.

    The parameters signal_range and fit_ranges are (NumPy) arrays specifying abcissa ranges, of the form [start_x, end_x].
    The former specifies the single range over which the signal of interest occurs, while the latter can specify multiple ranges
    over which to fit the desired polynomial background model, distributed over the second dimension of the fit_ranges array.
    Only a single row in fit_ranges is required, in which case the background will be extrapolated over the signal range and subtracted.
    If additional rows are supplied, then the background will be interpolated or extrapolated, depending on the range relationships.

    The order of the polynomial model function is specified by the polynomial_order parameter.
    The fit_log_data and fit_log_x parameters specify whether the corresponding axes of the spectral data should be transformed to
    logarithmic scales before doing the background fit.  This is to support exponential, Gaussian, and power-law background models.
    """

    # check shape and validity of data array
    data_dimension_count = len(data.shape)
    assert data_dimension_count < 3

    # extract spectral intensity (ordinate) and x (abcissa) arrays
    have_x_array = (data_dimension_count == 2)
    if have_x_array:
        assert min(data.shape[0], data.shape[1]) == 2
        if data.shape[0] == 2:
            x_values, data_values = data
        else:
            x_values = data[:, 0]
            data_values = data[:, 1]
    else:
        x_values = None
        data_values = data

    data_size = data_values.shape[0]
    assert data_size > polynomial_order + 3

    # check validity of abcissa value inputs
    if have_x_array:
        min_x = x_values.min()
        max_x = x_values.max()
    else:
        min_x = first_x
        max_x = first_x + data_size * delta_x
    assert max_x > min_x

    # check shape and validity of fit_ranges array
    range_dimension_count = len(fit_ranges.shape)
    assert range_dimension_count < 3
    assert fit_ranges.shape[range_dimension_count - 1] == 2

    # distill the fit ranges so that they are ordered, consolidated, and non-overlapping
    fit_ranges_clean = numpy.atleast_2d(numpy.sort(fit_ranges))
    fit_range_order = numpy.argsort(fit_ranges_clean, 0)[:, 0]
    fit_ranges_clean = fit_ranges_clean[fit_range_order]
    range_index = 1
    while range_index < fit_ranges_clean.shape[0]:
        if fit_ranges_clean[range_index, 0] > fit_ranges_clean[range_index - 1, 1]:
            range_index += 1
        else:
            fit_ranges_clean[range_index - 1, 1] = fit_ranges_clean[range_index, 1]
            fit_ranges_clean = numpy.delete(fit_ranges_clean, range_index, 0)

    # check validity of fit_ranges_clean array with respect to passed-in data range
    range_index = 0
    range_count = fit_ranges_clean.shape[0]
    while range_index < range_count:
        assert fit_ranges_clean[range_index, 0] >= min_x and fit_ranges_clean[range_index, 1] <= max_x
        range_index += 1

    # compile x and y arrays over fit ranges for input to the polynomial background fit
    if have_x_array:
        fit_range_selector = x_values >= fit_ranges_clean[0, 0] and x_values <= fit_ranges_clean[0, 1]
        range_index = 1
        while range_index < range_count:
            fit_range_selector = fit_range_selector or (x_values >= fit_ranges_clean[range_index, 0] and x_values <= fit_ranges_clean[range_index, 1])
            range_index += 1

        x_fit_values = x_values.compress(fit_range_selector)
        y_fit_values = data_values.compress(fit_range_selector)
    else:
        fit_start_chan = round((fit_ranges_clean[0, 0] - min_x) / delta_x)
        fit_end_chan = round((fit_ranges_clean[0, 1] - min_x) / delta_x) + 1
        x_fit_values = first_x + delta_x * numpy.arange(fit_start_chan, fit_end_chan)
        y_fit_values = data_values[fit_start_chan : fit_end_chan]

        range_index = 1
        while range_index < range_count:
            fit_start_chan = round((fit_ranges_clean[range_index, 0] - min_x) / delta_x)
            fit_end_chan = round((fit_ranges_clean[range_index, 1] - min_x) / delta_x) + 1
            numpy.append(x_fit_values, first_x + delta_x * numpy.arange(fit_start_chan, fit_end_chan))
            numpy.append(y_fit_values, data_values[fit_start_chan : fit_end_chan])
            range_index += 1

    # generate a polynomial fit over the fit ranges
    background_fit = CurveFitting.PolynomialFit1D(y_fit_values, x_fit_values, 0, 1, polynomial_order, fit_log_data, fit_log_x)

    # check shape and validity of signal_range array with respect to passed-in data range
    signal_range_clean = numpy.sort(signal_range).flatten()
    assert signal_range_clean.shape[0] == 2
    assert signal_range_clean[0] >= min_x and signal_range_clean[1] <= max_x

    # compute background model and subtract from data over contiguous union of fit and signal ranges
    bkgd_range = numpy.array([min(fit_ranges_clean[0, 0], signal_range_clean[0]), max(fit_ranges_clean[range_count - 1, 1], signal_range_clean[1])])

    # compute the net signal
    if have_x_array:
        in_bkgd_range = x_values >= bkgd_range[0] and x_values <= bkgd_range[1]
        bkgd_x_values = numpy.where(in_bkgd_range, x_values, bkgd_range[0])
        bkgd_fit = background_fit.compute_fit_for_values(bkgd_x_values)
        net_signal = numpy.where(in_bkgd_range, data_values - bkgd_fit, 0)
    else:
        bkgd_start_chan = round((bkgd_range[0] - min_x) / delta_x)
        bkgd_end_chan = round((bkgd_range[1] - min_x) / delta_x) + 1
        bkgd_fit = background_fit.compute_fit_for_range(bkgd_range, bkgd_end_chan - bkgd_start_chan)
        net_signal = numpy.zeros_like(data_values)
        net_signal[bkgd_start_chan : bkgd_end_chan] = data_values[bkgd_start_chan : bkgd_end_chan] - bkgd_fit

    return net_signal


def extract_signal_from_polynomial_background(data_and_metadata, signal_range, fit_ranges, first_x = 0.0, delta_x = 1.0,
                                                polynomial_order = 1, fit_log_data = False, fit_log_x = False):
    signal_range = numpy.asarray(signal_range) * data_and_metadata.data_shape[0]
    fit_ranges = numpy.asarray(fit_ranges) * data_and_metadata.data_shape[0]
    def data_fn():
        return extract_signal_from_polynomial_background_data(data_and_metadata.data, signal_range, fit_ranges, first_x, delta_x, polynomial_order, fit_log_data, fit_log_x)
    return DataAndMetadata.DataAndMetadata(data_fn, data_and_metadata.data_shape_and_dtype, data_and_metadata.intensity_calibration, data_and_metadata.dimensional_calibrations)


def subtract_linear_background(data_and_metadata, fit_range):
    fit_range = (numpy.asarray(fit_range) * data_and_metadata.data_shape[0]).astype(numpy.int)
    def data_fn():
        y = data_and_metadata.data[range(*fit_range)]
        A = numpy.vstack([numpy.arange(len(y)), numpy.ones((len(y), ))]).T
        m, c = numpy.linalg.lstsq(A, y)[0]
        return data_and_metadata.data - (numpy.arange(data_and_metadata.data_shape[-1]) * m + c)
    return DataAndMetadata.DataAndMetadata(data_fn, data_and_metadata.data_shape_and_dtype, data_and_metadata.intensity_calibration, data_and_metadata.dimensional_calibrations)


def generalized_oscillator_strength(energy_loss_eV: float, momentum_transfer_au: float,
                                    atomic_number: int, shell_number: int, subshell_index: int) -> numpy.ndarray:
    """Return the generalized oscillator strength as an ndarray.

    energy is in eV.
    scattering_angle is in radians.

    The 0-axis is in units of eV
    The 1-axis is in units of Phi * Phi / scattering angle
    The intensity is in units of nm * nm
    """
    pass


def partial_cross_section_nm2(beam_energy: float, energy_loss_start_eV: float, energy_loss_end_eV: float,
                              convergence_angle: float, collection_angle: float,
                              atomic_number: int, shell_number: int, subshell_index: int) -> float:
    """Returns the partial cross section.

    Uses generalized oscillator strength function.

    beam_energy is in eV.
    energy_start, energy_end are in eV.
    convergence_angle is in radians.
    collection_angle is in radians.

    The return value units are nm * nm.
    """
    pass


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


def edge_onset_energy_eV(atomic_number: int, shell_number: int, subshell_index: int) -> float:
    """Return the electron binding energy for the given edge.

    Return value is in eV.
    """
    pass


def edges_near_energy_eV(energy_loss_eV: float, energy_loss_delta_eV: float) -> list:
    """Return a list of edges near the energy_loss.
    """
    pass


# register functions
Context.registered_functions["extract_signal_from_polynomial_background"] = extract_signal_from_polynomial_background
Context.registered_functions["subtract_linear_background"] = subtract_linear_background
